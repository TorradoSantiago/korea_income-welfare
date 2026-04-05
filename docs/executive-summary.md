# Executive Summary

## Metadata and original objective

The original notebook began with a strong analytical setup: it described the full welfare dataset, clarified the meaning of the variables, and framed the core objective as understanding how education affects income in Korea. That structure is preserved here because it gives the repository a clearer purpose than a generic modeling exercise.

The original objective was simple and strong: determine how education affects a person's income and use that relationship to generate useful insight for individuals, institutions, firms, and policymakers.

## Research question and hypotheses

- `H0`: education does not have a meaningful effect on income in this sample.
- `H1`: education has a meaningful positive effect on income, and different education levels are associated with different income ranges.
- `H2`: a broader socioeconomic model should materially outperform an education-only baseline.
- `H3`: the return to education should vary across the income distribution rather than remaining constant.

## Commercial and policy context

The original delivery did something important that is often missing from portfolio work: it explained why the question matters. Individuals want to know whether investing in education pays off. Educational institutions need to understand whether their programs are aligned with labor-market outcomes. Policymakers want evidence on how educational attainment relates to earnings and social mobility. That context remains central to the repository because it makes the analysis more than a technical benchmark.

## Analytical approach

The upgraded repository now combines three layers:

1. A descriptive layer that profiles `89,935` valid observations and shows how income varies by education, region, and time.
2. A tuned ML benchmark that compares five out-of-sample models and finds a best-performing random forest with `R^2 = 0.645`.
3. An econometric layer that estimates a semilog OLS model and quantile regressions to test whether education premiums are heterogeneous across the income distribution.

## What the evidence shows

- Income rises steadily across the education ladder, from roughly `999 USD` on average for people with no formal education to `5,480 USD` for doctorate holders in the trimmed sample.
- Seoul leads the regional distribution with a median income of roughly `2,884 USD`.
- The education-only linear regression reaches `R^2 = 0.294`.
- The multivariable linear model improves to `R^2 = 0.545`.
- Gradient boosting reaches `R^2 = 0.603`.
- The tuned random forest reaches `R^2 = 0.645` and becomes the strongest predictive benchmark.
- The semilog OLS model estimates a `12.47%` income premium for each additional education step.
- The quantile regressions show that the education premium rises across the distribution: `11.46%` at the 25th percentile, `13.84%` at the median, and `14.17%` at the 75th percentile.

## Interpretation

The ML layer and the econometric layer tell a coherent story. The tuned random forest shows that household size, age, education, and year all matter for prediction. The econometric layer then explains what that means: education is a meaningful socioeconomic signal, but its payoff is not constant. It grows in the upper part of the income distribution.

## Main conclusion

The project is strongest when it is read as both a predictive and an interpretive case. The tuned model proves that a richer socioeconomic specification materially improves prediction, while the quantile regressions show that education has heterogeneous returns. That combination is much closer to a real applied-economics or analytics-consulting case than a single benchmark notebook.

## Original delivery preservation

The original Spanish notebook and its theoretical narrative are explicitly preserved in `archive/original_delivery/`. That archive matters because it keeps the first academic framing visible and usable, instead of replacing it with a thinner portfolio summary.
