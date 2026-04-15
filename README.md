# Korea Income and Welfare
**Korean welfare panel microdata — Python**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

---

Does education still carry a measurable income premium once household structure, age dynamics, regional sorting, gender, and survey wave are held constant? This project answers that question using roughly 90,000 person-year observations from the Korea Welfare Panel Study (KoWePS).

The analysis combines descriptive EDA, non-parametric hypothesis tests, a semilog OLS earnings equation, quantile regression, and a five-model predictive benchmarking suite. The result is a richer story than a single R² score: machine learning genuinely outperforms linear regression here (unlike in a typical cross-section), because the panel structure and household-level interactions contain nonlinear signal that OLS cannot capture — and quantile regression shows that the education premium is not a flat scalar but rises through the income distribution.

**Three hypotheses, three confirmations:**

| Hypothesis | Finding |
|---|---|
| H1 — Education premium | OLS: **+12.47%** per education step, highly significant after all controls |
| H2 — Distributional heterogeneity | Premium rises from **11.46%** (Q25) → **14.17%** (Q75) — consistent with labor-market sorting |
| H3 — ML adds value beyond OLS | Tuned Random Forest: **R² = 0.645** vs OLS **R² = 0.545** (holdout) |

> For the full technical writeup — econometric specification, diagnostic tests, and quantile regression discussion — see [`docs/korea-income-welfare-brief.pdf`](docs/korea-income-welfare-brief.pdf).

---

## Structure

```
korea_income-welfare/
├── income_welfare_modeling.ipynb      ← start here (narrative walkthrough)
├── analysis_final.py                  ← regenerates all 14 figures + 12 tables
├── Korea_Income_and_Welfare.csv       ← raw data (92,857 records)
├── requirements.txt
├── outputs/
│   ├── figures/                       ← 14 publication-ready charts
│   └── tables/                        ← 12 CSV outputs
└── docs/
    ├── korea-income-welfare-brief.tex ← LaTeX source
    └── korea-income-welfare-brief.pdf ← compiled technical brief
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook income_welfare_modeling.ipynb
# or: python analysis_final.py
```

---

## Methods

| Step | What | Why |
|------|------|-----|
| Descriptive EDA | Income summaries and charts by education, region, year, and gender | Establish unconditional patterns before modeling |
| Non-parametric tests | Kruskal-Wallis, Spearman ρ, Mann-Whitney U | Income is right-skewed; parametric ANOVA is not appropriate |
| Predictive benchmark | Education-only OLS → Elastic Net → Gradient Boosting → tuned Random Forest | Show the incremental value of each modeling layer |
| Semilog OLS | Log(income) ~ education + age polynomial + household size + year + region/gender/marriage/religion dummies | Estimate interpretable average education premium with HC3 robust errors |
| Diagnostics | Breusch-Pagan, RESET, VIF | Validate heteroskedasticity correction, flag nonlinearity, check collinearity |
| Quantile regression | Re-estimate OLS at Q25, Q50, Q75 | Test whether education returns are constant or rise through the distribution |

---

## Econometric specification

The core model is a semilog earnings equation:

```
log(income_usd) ~ education_level
               + age_centered + age_centered²
               + family_member
               + year
               + C(region) + C(gender) + C(marriage) + C(religion)
```

Age is centered before squaring to reduce multicollinearity (VIF strategy). Heteroskedasticity-consistent HC3 standard errors are used throughout — the Breusch-Pagan test rejects homoskedasticity. The RESET test rejects the simple linear form, which is why the analysis pairs OLS with machine learning and quantile regression rather than stopping at a single coefficient table.

**Why this differs from a classic Mincer specification:** The KoWePS data does not include continuous years of schooling or hours worked, so the standard `experience = age - schooling - 6` construction is not feasible. Education enters as a nine-point ordinal index; age enters directly as a centered quadratic life-cycle term.

---

## Key figures

| Figure | Content |
|--------|---------|
| fig01 | Average income by education level |
| fig02 | Income distribution (box plots) by education |
| fig03 | Median income by region |
| fig04 | Income trend across survey waves (median + mean) |
| fig05 | Raw gender income gap |
| fig06 | Gender gap by education level |
| fig07 | Hypothesis tests (−log₁₀ p-values) |
| fig08 | OLS coefficient estimates with 95% HC3 CI |
| fig09 | OLS diagnostics (Q-Q plot + residuals vs fitted) |
| fig10 | Education premium: OLS mean vs quantile regressions |
| fig11 | Out-of-sample R² model comparison |
| fig12 | 5-fold cross-validated R² with error bars |
| fig13 | Feature importance — tuned Random Forest |
| fig14 | Actual vs predicted income (best model) |

---

**Author:** Santiago Torrado — Applied economics, data analysis, public policy
