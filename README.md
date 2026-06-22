# Adult-Income-Analysis
Analysing how various features in the Adult dataset contribute to an individual's income level.

## Reproducibility
1. Create and activate a **Python 3.11** environment.
2. Install the pinned dependencies with `pip install -r requirements.txt`.
3. Keep `adult.csv` in the repository root.
4. Open `Project_4__Part1&2(Core)_Gladys_Babirye.ipynb`, restart the `python3` kernel, and run all cells from top to bottom.
5. The notebook reads data through the local `dw_use` helper and writes reproducible artifacts to `outputs/` and `figures/`. Each saved artifact also gets a `.provenance.json` sidecar with runtime and source metadata.

- Visuals of the features: educational-num & marital-status which were among the top 10 features from my permutation importances.produce. These explanatory visualizations show the relationship between the feature and the target: Income.


**Income vs educational-num**

  
![income vs educational-num](https://github.com/gladysbabs/Adult-Income-Analysis/assets/162020572/7053c2c2-1766-4253-9938-2f69ea2cd4d3)

This plot reveals that individuals that studied for between 9-12 years, managed to secure an increase in their income levels.

**Income vs marital-status**


![income vs marital-status](https://github.com/gladysbabs/Adult-Income-Analysis/assets/162020572/a8430f26-7721-49f0-aee5-da57e1a9d9dc)

The married vs income multiplot does not reveal alot on how marital-status is a contributing factor to an individual's income. However, I can only tell that Married individuals are predominant in securing an income.
