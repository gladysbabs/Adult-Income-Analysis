from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping


ALLOWED_ARTIFACT_DIRS = ("artifacts", "figures", "outputs")
REPO_ROOT_MARKERS = ("README.md", "requirements.txt")


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
        if all((candidate / marker).exists() for marker in REPO_ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        "Could not find the repository root. Open the notebook from this repository "
        "or run it from a child directory that contains the project files."
    )


def resolve_repo_path(
    path: str | Path,
    *,
    repo_root: Path | None = None,
    must_exist: bool = True,
) -> Path:
    root = repo_root.resolve() if repo_root else find_repo_root()
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{path!s} resolves outside the repository root ({root}). "
            "Use repository-local input files for reproducible runs."
        ) from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Could not find {path!s}. Expected it at {resolved}. "
            "Keep the input file in the repository and run the notebook from this repo."
        )

    return resolved


def dw_use(path: str | Path, *, repo_root: Path | None = None, **kwargs: Any) -> Any:
    resolved = resolve_repo_path(path, repo_root=repo_root, must_exist=True)
    suffix = resolved.suffix.lower()

    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(resolved, **kwargs)
    if suffix == ".json":
        return json.loads(resolved.read_text(encoding="utf-8"))
    if suffix in {".txt", ".md"}:
        return resolved.read_text(encoding="utf-8")

    raise ValueError(
        f"dw_use does not support '{suffix}' files yet. "
        "Add the format explicitly before using it in the notebook."
    )


def dw_save(
    obj: Any,
    path: str | Path,
    *,
    repo_root: Path | None = None,
    overwrite: bool = False,
    metadata: Mapping[str, Any] | None = None,
    notebook_path: str | Path | None = None,
    source_paths: Iterable[str | Path] | None = None,
    provenance: bool = True,
    **kwargs: Any,
) -> Path:
    root = repo_root.resolve() if repo_root else find_repo_root()
    resolved = _resolve_artifact_path(path, repo_root=root)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    if resolved.exists() and not overwrite:
        raise FileExistsError(
            f"{resolved} already exists. Pass overwrite=True to replace the artifact."
        )

    suffix = resolved.suffix.lower()
    if _is_dataframe(obj):
        if suffix != ".csv":
            raise ValueError("DataFrame artifacts must be saved as .csv in this repository.")
        obj.to_csv(resolved, index=kwargs.pop("index", False), **kwargs)
    elif _is_matplotlib_figure(obj):
        if suffix not in {".png", ".svg", ".pdf"}:
            raise ValueError("Figure artifacts must use .png, .svg, or .pdf.")
        obj.savefig(resolved, bbox_inches=kwargs.pop("bbox_inches", "tight"), **kwargs)
    elif suffix == ".json":
        resolved.write_text(
            json.dumps(obj, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
    elif suffix in {".txt", ".md"}:
        resolved.write_text(str(obj), encoding="utf-8")
    else:
        raise ValueError(
            f"dw_save does not support '{suffix}' outputs yet. "
            "Add the format explicitly before using it in the notebook."
        )

    if provenance:
        provenance_path = Path(f"{resolved}.provenance.json")
        resolved_source_paths = [
            resolve_repo_path(source_path, repo_root=root, must_exist=False)
            for source_path in (source_paths or [])
        ]
        provenance_payload = {
            "artifact_path": str(resolved),
            "artifact_path_relative_to_repo": str(resolved.relative_to(root)),
            "sha256": _sha256(resolved),
            "size_bytes": resolved.stat().st_size,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "notebook_path": str(resolve_repo_path(notebook_path, repo_root=root, must_exist=False))
            if notebook_path
            else None,
            "source_paths": [str(_relative_to_repo(path, repo_root=root)) for path in resolved_source_paths],
            "source_files": [
                {
                    "path": str(path),
                    "path_relative_to_repo": str(_relative_to_repo(path, repo_root=root)),
                    "exists": path.exists(),
                    "sha256": _sha256(path) if path.exists() and path.is_file() else None,
                }
                for path in resolved_source_paths
            ],
            "runtime": collect_runtime_snapshot(
                repo_root=root,
                data_path=resolved_source_paths[0] if resolved_source_paths else None,
            ),
            "metadata": dict(metadata or {}),
        }
        provenance_path.write_text(
            json.dumps(provenance_payload, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )

    return resolved


def collect_runtime_snapshot(
    *,
    repo_root: Path | None = None,
    notebook_path: str | Path | None = None,
    data_path: str | Path | None = None,
    packages: Iterable[str] = (
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "imbalanced-learn",
        "collinearity",
        "notebook",
        "ipykernel",
    ),
) -> dict[str, Any]:
    root = repo_root.resolve() if repo_root else find_repo_root()

    snapshot: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repo_root": str(root),
        "git_branch": _git_value(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "packages": _package_versions(packages),
    }

    if notebook_path:
        snapshot["notebook_path"] = str(
            resolve_repo_path(notebook_path, repo_root=root, must_exist=False)
        )
    if data_path:
        resolved_data_path = resolve_repo_path(data_path, repo_root=root, must_exist=True)
        snapshot["data_path"] = str(resolved_data_path)
        snapshot["data_sha256"] = _sha256(resolved_data_path)

    return snapshot


def _resolve_artifact_path(path: str | Path, *, repo_root: Path) -> Path:
    raw_path = Path(path).expanduser()
    resolved = raw_path.resolve() if raw_path.is_absolute() else (repo_root / raw_path).resolve()

    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Artifacts must be written inside this repository.") from exc

    if not relative.parts or relative.parts[0] not in ALLOWED_ARTIFACT_DIRS:
        allowed = ", ".join(ALLOWED_ARTIFACT_DIRS)
        raise ValueError(f"Artifacts must be saved under one of: {allowed}.")

    return resolved


def _relative_to_repo(path: Path, *, repo_root: Path) -> Path:
    return path.relative_to(repo_root)


def _package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_value(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_dataframe(obj: Any) -> bool:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return False
    return isinstance(obj, pd.DataFrame)


def _is_matplotlib_figure(obj: Any) -> bool:
    try:
        from matplotlib.figure import Figure
    except ModuleNotFoundError:
        return False
    return isinstance(obj, Figure)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
