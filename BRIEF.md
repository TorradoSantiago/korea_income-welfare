# Executive Brief

## In one sentence

Applied income-analysis case on Korean welfare microdata, now strengthened with formal distribution tests, cross-validated ML benchmarking, centered-age econometrics, and quantile regression to show heterogeneous education returns.

## What problem it addresses

The project asks whether education predicts income, how much prediction improves once household and demographic variables are added, and whether the payoff to education is the same at lower-income and higher-income parts of the distribution.

## Why it is stronger now

- The descriptive layer now has explicit statistical tests behind it.
- The OLS model includes a centered age polynomial and diagnostics.
- The predictive comparison is no longer a single holdout exercise; it now includes five-fold cross-validation.
- Quantile regression turns the project from a generic prediction task into a more economic story about heterogeneous returns.

## What a non-technical reader can see

- A large, policy-relevant dataset.
- A clear positive link between education and income.
- Stronger predictive performance once family structure, region, gender, religion, and year are included.
- A richer conclusion: education seems to pay off more in higher-income parts of the distribution.

## Technical highlights

- Kruskal-Wallis across education levels: overwhelmingly significant.
- Spearman education-income association: `rho = 0.609`.
- OLS education premium: `12.47%` per education step.
- Quantile premiums: `11.46%`, `13.84%`, and `14.17%` from lower to upper quantiles.
- Best holdout ML model: Tuned Random Forest, `R^2 = 0.645`.
- Best cross-validated ML model: Tuned Random Forest, `CV R^2 = 0.649`.

## Current conclusion

The project now supports two conclusions at once: education has a strong average relationship with income, and that relationship is not uniform across the distribution. The ML layer improves prediction materially, while the econometric layer explains what that predictive improvement means.

## Best next step

Add a segmented appendix by gender or region to test whether education premiums and model gains differ across subpopulations.
