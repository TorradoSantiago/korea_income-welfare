# Korea Income and Welfare
**Korean welfare microdata - Python**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

---

This project studies a simple question with real labor-market relevance: **how much does education still matter for income once we account for household structure, demographics, region, and time?**

Using Korean welfare microdata, the analysis combines descriptive EDA, non-parametric tests, predictive benchmarking, semilog OLS with robust inference, and quantile regression. The result is a stronger portfolio story than a single model score: machine learning improves prediction, but the econometric layer explains where the education premium comes from and why it is not constant across the income distribution.

**Four findings stand out:**

- Education carries an average OLS premium of roughly **12.47%** per step
- The education premium rises from **11.46%** at the 25th percentile to **14.17%** at the 75th percentile
- A tuned random forest is the strongest predictive model with **R^2 = 0.645** on holdout data and **CV R^2 = 0.649**
- Household size, age, education, and survey year explain most of the predictive gain

> For the full technical writeup, model summary, and interpretation, see [`docs/korea-income-welfare-brief.pdf`](docs/korea-income-welfare-brief.pdf).

---

## Structure

```
|-- income_welfare_modeling.ipynb      -> start here
|-- analysis_final.py                  -> regenerates figures and tables
|-- Korea_Income_and_Welfare.csv       -> raw data
|-- outputs/figures/                   -> publication-ready charts
|-- outputs/tables/                    -> model outputs as CSV
\-- docs/korea-income-welfare-brief.*  -> LaTeX brief and compiled PDF
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook income_welfare_modeling.ipynb
# or: python analysis_final.py
```

## Methods

| Step | What | Why |
|------|------|-----|
| Descriptive EDA | Income summaries by education, region, and year | Establish the structure of the sample before modeling |
| Statistical tests | Kruskal-Wallis, Spearman, Mann-Whitney U | Confirm the descriptive patterns formally |
| Predictive benchmark | Education-only OLS, multivariable linear regression, Elastic Net, Gradient Boosting, tuned Random Forest | Compare simple and richer predictive baselines |
| Econometric model | Semilog OLS with centered age polynomial and HC3 robust errors | Estimate an interpretable average education premium |
| Diagnostics | Breusch-Pagan, RESET, and VIF | Check heteroskedasticity, misspecification, and collinearity |
| Distributional layer | Quantile regression at the 25th, 50th, and 75th percentiles | Test whether education returns vary across the income distribution |

**Author:** Santiago Torrado - Applied economics, data analysis, public policy
