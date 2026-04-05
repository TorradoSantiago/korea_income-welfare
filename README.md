# Korea Income & Welfare

Portfolio-grade case study on Korean welfare microdata that now combines machine-learning benchmarking with an econometric earnings analysis. The repository still answers the original question about education and income, but it now does so with two complementary lenses: predictive performance and interpretable coefficient-based inference.

If you want a short, non-technical explanation first, start with `BRIEF.md`.

## Executive Summary

- Dataset: `92,857` survey records and `14` raw variables.
- Valid analysis sample: `89,935` records.
- Trimmed modeling sample used for ML and econometrics: `86,313` records.
- Best predictive benchmark: Multivariable Random Forest (`MAE = 908.88`, `RMSE = 1279.94`, `R^2 = 0.628`).
- Econometric layer:
  - each education step is associated with a `12.47%` income premium in the OLS semilog model;
  - the estimated education premium rises across the income distribution, from `11.46%` at the 25th percentile to `14.17%` at the 75th percentile.
- Senior takeaway: education matters, but its payoff is heterogeneous and becomes more pronounced in the upper half of the income distribution.

## Business and Policy Relevance

Understanding the education-income relationship is useful for:

- public policy teams designing workforce and welfare interventions;
- education providers evaluating the labor-market payoff of credentials;
- analysts building explainable socioeconomic benchmarks;
- consulting teams translating socioeconomic data into decision-ready narratives.

## Analytical Question

How much of individual income can education explain, how much additional signal appears in a multivariable machine-learning setup, and do returns to education stay constant across the income distribution?

## Analytical Workflow

1. Load and profile the welfare microdata.
2. Map coded fields into readable labels and convert income to USD.
3. Filter invalid observations and trim target outliers with the IQR rule.
4. Benchmark three predictive models on the same target:
   - Education-only Linear Regression
   - Multivariable Linear Regression
   - Multivariable Random Forest Regressor
5. Estimate a semilog OLS model for income.
6. Estimate quantile regressions at the 25th, 50th, and 75th percentiles.
7. Export reusable tables and charts for portfolio, interviews, and LinkedIn content.

## Key Findings

- Income rises steadily across the education ladder, which supports the central hypothesis.
- The multivariable random forest remains the strongest predictive benchmark, confirming that household structure, age, time, and region matter alongside education.
- In the econometric layer, one additional education step is associated with a `12.47%` increase in income on average.
- The return to education is not constant across the distribution: it is lower in the 25th percentile (`11.46%`) and higher in the 75th percentile (`14.17%`).
- That heterogeneity matters. It suggests education is not just shifting the mean; it is linked to larger payoffs in better-paid segments of the sample.

## Model Comparison

| Model | MAE (USD) | RMSE (USD) | R^2 |
| --- | ---: | ---: | ---: |
| Education-only linear regression | 1345.40 | 1762.56 | 0.294 |
| Multivariable linear regression | 1049.83 | 1414.49 | 0.545 |
| Multivariable random forest | 908.88 | 1279.94 | 0.628 |

## Repository Structure

- `income_welfare_modeling.ipynb`: main notebook companion
- `archive/income_welfare_modeling_legacy_es.ipynb`: preserved original Spanish notebook
- `Korea_Income_and_Welfare.csv`: source dataset
- `BRIEF.md`: short explanation for non-technical readers
- `docs/executive-summary.md`: polished findings and portfolio framing
- `scripts/income_welfare_analysis.py`: reproducible ML plus econometric pipeline
- `outputs/tables/model_comparison.csv`: predictive benchmark table
- `outputs/tables/feature_importance.csv`: random-forest feature importances
- `outputs/tables/econometric_ols_summary.csv`: key semilog OLS coefficients
- `outputs/tables/quantile_regression_summary.csv`: quantile-regression results for the education premium
- `outputs/figures/top_feature_importance.png`: random-forest explanation graphic
- `outputs/figures/education_premium_by_quantile.png`: education-premium comparison across mean and quantile models
- `docs/korea-income-welfare-presentation.html`: print-ready executive brief
- `docs/korea-income-welfare-presentation.pdf`: portfolio-ready presentation asset

## Run Locally

```bash
pip install -r requirements.txt
python scripts/income_welfare_analysis.py
jupyter notebook "income_welfare_modeling.ipynb"
```

## Limitations

- The encoded survey categories should still be read as directional signals, not causal evidence.
- The education variable is ordinal rather than a direct count of schooling years.
- The quantile results show heterogeneity, but they do not identify a causal mechanism for why returns differ across the distribution.

## Why it Works in a Portfolio

This project now demonstrates two complementary strengths:

- the ability to build and compare predictive models honestly;
- the ability to step back and ask a more econometric question about heterogeneous returns.

That mix makes the repository stronger for analytics, applied economics, consulting, and data strategy conversations than a pure ML benchmark alone.
