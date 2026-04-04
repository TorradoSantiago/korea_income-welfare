# Korea Income & Welfare

A portfolio-grade machine learning case study on the relationship between education and income in Korean welfare microdata. The repository combines exploratory analysis, feature mapping, and model benchmarking to answer a simple but policy-relevant question: how much signal does education carry when predicting individual income?

If you want a short, non-technical explanation first, start with `BRIEF.md`.

## Executive Summary

- Dataset: 92,857 survey records and 14 raw variables.
- Valid records in the upgraded analysis layer: 89,935.
- Main analytical question: how much signal does education provide once the benchmark becomes multivariable?
- Best benchmark in the upgraded setup: Multivariable Random Forest (`MAE = 908.88`, `RMSE = 1279.94`, `R^2 = 0.628`).
- Senior takeaway: education matters, but household structure, age, and regional context materially improve explanatory power.

## Business and Policy Relevance

Understanding the education-income relationship is useful for:

- public policy teams designing workforce and welfare interventions;
- education providers evaluating the labor-market payoff of credentials;
- analysts building explainable socioeconomic benchmarks.

## Hypothesis

- `H0`: education level does not have a meaningful relationship with income in this sample.
- `H1`: education level has a meaningful positive relationship with income, with higher education associated with higher earnings.

## Analytical Workflow

1. Load and profile the welfare microdata.
2. Map coded categorical fields into readable labels.
3. Convert income from thousand KRW to USD and derive age.
4. Remove sparse columns and filter outliers with the IQR rule.
5. Benchmark three models on the same target:
   - Education-only Linear Regression
   - Multivariable Linear Regression
   - Multivariable Random Forest Regressor

## Key Findings

- Average income rises across education tiers, supporting the central hypothesis.
- The upgraded random-forest model shows that `family_member`, `education_level`, `age`, and `year` carry most of the predictive signal.
- Seoul leads the regional distribution in the cleaned sample.
- The multivariable benchmark materially outperforms the original one-feature formulation.

## Model Comparison

| Model | MAE (USD) | RMSE (USD) | R^2 |
| --- | ---: | ---: | ---: |
| Education-only linear regression | 1345.40 | 1762.56 | 0.294 |
| Multivariable linear regression | 1049.83 | 1414.49 | 0.545 |
| Multivariable random forest | 908.88 | 1279.94 | 0.628 |

## Repository Structure

- `income_welfare_modeling.ipynb`: full exploratory and modeling workflow.
- `archive/income_welfare_modeling_legacy_es.ipynb`: preserved original Spanish exploratory notebook.
- `Korea_Income_and_Welfare.csv`: source dataset.
- `scripts/income_welfare_analysis.py`: reproducible analysis pipeline in English.
- `docs/executive-summary.md`: polished hypotheses, findings, and conclusions.
- `docs/korea-income-welfare-presentation.html`: print-ready executive brief.
- `docs/korea-income-welfare-presentation.pdf`: portfolio-ready presentation asset.
- `outputs/tables/`: reusable benchmark and summary tables.
- `outputs/figures/`: reusable charts for portfolio and LinkedIn use.

## Run Locally

```bash
pip install -r requirements.txt
python scripts/income_welfare_analysis.py
jupyter notebook "income_welfare_modeling.ipynb"
```

## Limitations

- Encoded categories should still be read as directional signals, not causal evidence.
- The notebook itself has historical value, but the canonical portfolio layer is now the English script and PDF brief.
- A stronger next iteration would add model explainability by segment, not just global feature importance.

## Why it Works in a Portfolio

This project is a strong portfolio case because it links public-interest data, business framing, and interpretable model benchmarking. It reads well for roles in data analysis, applied economics, analytics consulting, and policy tech.
