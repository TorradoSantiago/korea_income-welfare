# Executive Summary

## Project purpose

This repository studies how education relates to income in Korean welfare microdata. The original delivery already made an important conceptual move: it treated the education-income relationship as a meaningful social and economic question, not merely as a predictive convenience. The upgraded version preserves that logic while adding more technical discipline and more convincing outputs for a professional audience.

## Conceptual base

The project is built on a simple but important premise: education matters economically, but its effect should not be assumed to be uniform. Education can capture formal human capital, credentialing, access to occupations, and sorting into stronger labor-market trajectories. That means a good project should ask at least three questions:

1. Is education associated with income at all?
2. How much better can income be predicted when broader socioeconomic variables are added?
3. Does the education premium stay constant across the income distribution?

## Data and sample

- Raw dataset: `92,857` records.
- Valid analysis sample after cleaning: `89,935` records.
- Trimmed modeling sample for predictive and econometric layers: `86,313` records.

## Methodological upgrade

### 1. Descriptive layer

Income is summarized by education level, region, and time. This keeps the project readable and preserves the original intuition that the best modeling work starts with a coherent empirical story.

### 2. Statistical testing

The upgraded version now tests whether the descriptive relationships are real:

- Kruskal-Wallis confirms sharp differences in income distributions across education levels.
- Spearman correlation confirms a strong positive monotonic association between education and income.
- Mann-Whitney confirms that male and female income distributions still differ before controls.

### 3. Predictive benchmarking

The ML layer now compares five models:

- education-only linear regression,
- multivariable linear regression,
- Elastic Net,
- Gradient Boosting,
- tuned Random Forest.

Performance is evaluated both on a holdout split and through five-fold cross-validation. That makes the comparison much more defensible than a single split.

### 4. Econometric interpretation

The econometric layer estimates a semilog OLS model with a centered age polynomial and then extends the analysis using quantile regression. This is the key conceptual improvement: the project no longer only says that education predicts income. It now asks whether the payoff to education changes across the distribution.

## Main empirical findings

- Education distributions differ strongly in their observed income outcomes.
- The OLS model estimates a `12.47%` income premium per education step.
- The education premium is `11.46%` at the 25th percentile, `13.84%` at the median, and `14.17%` at the 75th percentile.
- The tuned random forest is the best predictive benchmark with `R^2 = 0.645` on the holdout split.
- The same model remains strongest under cross-validation with `CV R^2 = 0.649`.

## Diagnostic interpretation

The upgraded repository adds diagnostic evidence that improves credibility.

- Centering the age polynomial lowers VIF values to manageable levels.
- Heteroskedasticity is present, so robust standard errors are appropriate.
- The strong RESET result suggests that purely linear approximations do not fully exhaust the structure of the data, which helps justify keeping the ML layer alongside the coefficient-based analysis.
- The best predictive model identifies household size, age, education, and year as the dominant features.

## Theoretical contribution

The central analytical contribution is no longer just that education matters. The more interesting conclusion is that education seems to matter more at higher parts of the income distribution. That suggests that schooling interacts with labor-market sorting, occupational matching, or credential-based access to better-paying opportunities rather than producing a completely uniform premium.

## Conclusion

This is now a stronger portfolio project because it combines scale, transparency, predictive benchmarking, and interpretable econometrics. It keeps the clearer academic motivation of the original delivery while producing a more technical and more defensible professional case.
