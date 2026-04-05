# Korea Income & Welfare

## Executive summary

This repository studies how education relates to income using Korean welfare microdata and now includes both a tuned predictive benchmarking layer and a proper econometric extension. The updated workflow processes `89,935` valid records, trims the modeling sample to `86,313` observations, exports reusable figures, and compares model performance against semilog and quantile-regression estimates.

## Analytical question

How much of individual income can education explain, how much predictive power appears once age, household structure, time, and social attributes are included, and are the returns to education constant across the income distribution?

## Updated hypotheses

- `H1`: education has a meaningful positive relationship with income.
- `H2`: education is informative but incomplete, so a broader multivariable model should materially outperform an education-only benchmark.
- `H3`: the return to education is heterogeneous and should be larger in the upper part of the income distribution.
- `H4`: a tuned tree-based model should outperform untuned baselines without changing the substantive conclusion.

## What the upgraded analysis shows

- Income rises steadily across the education ladder, from roughly `999 USD` on average for people with no formal education to `5,480 USD` for doctorate holders in the trimmed sample.
- Seoul leads the regional distribution with a median income of roughly `2,884 USD`.
- The education-only linear regression reaches `R^2 = 0.294`, which confirms signal but leaves most variance unexplained.
- The multivariable linear model improves to `R^2 = 0.545`.
- Gradient boosting reaches `R^2 = 0.603`.
- The tuned random forest reaches `R^2 = 0.645`, making it the strongest predictive benchmark in the repository.
- The semilog OLS model estimates a `12.47%` income premium for each additional education step.
- The quantile regressions show that the education premium rises across the distribution: `11.46%` at the 25th percentile, `13.84%` at the median, and `14.17%` at the 75th percentile.

## Most important modeling lesson

The ML layer and the econometric layer now complement each other. The tuned random forest shows that household size, age, education, and year all carry major predictive signal. The quantile-regression layer sharpens that conclusion by showing that education does not pay off equally everywhere in the distribution. Its association with income is stronger in the better-paid segments of the sample.

## Improved conclusion

Education matters and should stay at the center of the story, but the stronger portfolio-quality conclusion is more nuanced: education is a meaningful socioeconomic signal, its payoff is heterogeneous, and a serious analysis benefits from both tuned predictive models and interpretable coefficient-based evidence. That makes the repository more credible for analytics, policy, and consulting positioning than a pure one-model notebook.

## Portfolio takeaway

This repository now demonstrates:

- a reproducible analysis pipeline in English;
- reusable charts, benchmark tables, and tuned hyperparameters;
- a bridge between machine learning and econometrics;
- a more senior interpretation of what the models can and cannot explain.
