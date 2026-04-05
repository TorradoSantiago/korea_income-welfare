# Korea Income & Welfare

Portfolio-grade case study on Korean welfare microdata that combines predictive benchmarking with an econometric earnings analysis. The repository still answers the original question about education and income, but now preserves the original theoretical framing alongside a much more technical public-facing layer.

Start with:
- `income_welfare_modeling.ipynb`
- `BRIEF.md`
- `docs/executive-summary.md`
- `docs/korea-income-welfare-presentation.pdf`
- `archive/original_delivery/original_theoretical_notes_es.md`

## Recovered original delivery

The original project submission is explicitly preserved in `archive/original_delivery/`:

- `Modelo Machine Learning.ipynb`: original notebook restored with its delivery filename.
- `README_original_2024.md`: earliest repository README recovered from history.
- `original_theoretical_notes_es.md`: extracted theoretical and methodological explanations from the Spanish notebook.
- `recovery_manifest.md`: provenance note explaining what was recovered and from which commits.

## Research question

The core question is whether education has measurable economic value in Korean welfare microdata, how stable that relationship remains after adding demographic and household controls, and whether the return to education is constant across the income distribution.

## Why the project matters theoretically

The original notebook framed education and income as a real social and economic decision problem. That remains the right starting point. Education is not only a predictive variable. It is a proxy for human capital, credentialing, access to opportunities, and differentiated life trajectories. If the relationship between education and income is both strong and heterogeneous, that matters to individuals, institutions, and policymakers.

## Analytical layers

1. Descriptive baseline
   - Income by education level, region, and year.

2. Formal distribution tests
   - Kruskal-Wallis across education levels.
   - Spearman education-income association.
   - Mann-Whitney gender comparison.

3. Predictive benchmarking
   - Education-only linear regression.
   - Multivariable linear regression.
   - Elastic Net.
   - Gradient Boosting.
   - Tuned Random Forest.
   - Five-fold cross-validation.

4. Econometric interpretation
   - Semilog OLS with centered age polynomial.
   - Quantile regression to test heterogeneous education returns.
   - Diagnostic tables for heteroskedasticity, specification, and collinearity.

## Core sample and results

- Raw dataset: `92,857` survey records.
- Valid analysis sample: `89,935` records.
- Trimmed modeling sample: `86,313` records.
- Best holdout model: Tuned Random Forest with `R^2 = 0.645`.
- Best cross-validated model: Tuned Random Forest with `CV R^2 = 0.649`.
- Average education premium in OLS: `12.47%` per education step.
- Education premium by quantile:
  - `11.46%` at the 25th percentile,
  - `13.84%` at the median,
  - `14.17%` at the 75th percentile.

## What the diagnostics add

The project is more credible now because it no longer stops at model performance.

- Education distributions differ sharply across income outcomes even before modeling.
- Centering the age polynomial lowers VIF values to roughly `1.03` to `1.96`, which stabilizes the nonlinear age profile.
- Heteroskedasticity is present, so the main OLS table is reported with robust errors.
- RESET strongly rejects the idea that a simple functional form is fully sufficient, which helps justify the complementary ML benchmark and the quantile-regression layer.

## Interpreting the ML layer honestly

The ML benchmark is strong, but it does not replace interpretation.

- The tuned random forest materially outperforms linear models for pure prediction.
- The feature-importance ranking shows that household size, age, education, and year explain most of the predictive gain.
- The econometric layer remains essential because it tells us how the education premium changes across the income distribution.

This is a better portfolio story than simply reporting the highest `R^2`.

## Theoretical proposal

The repository now supports a richer claim: education has a positive income premium on average, but that premium is not constant across the distribution. Returns appear larger in better-paid parts of the sample, which suggests that schooling interacts with labor-market sorting rather than operating as a uniform scalar reward.

## Conclusion

This project is now strong for analytics, consulting, and applied-economics positioning because it combines scale, benchmarking, tuning, coefficient interpretation, and distributional heterogeneity. It preserves the original theoretical motivation while making the empirical case more technical and more persuasive.
