# Korea Income & Welfare

Portfolio-grade case study on Korean welfare microdata that combines tuned machine-learning benchmarking with an econometric earnings analysis. The repository still answers the original question about education and income, but it now preserves the original theoretical framing alongside the upgraded portfolio layer.

If you want a short, non-technical explanation first, start with `BRIEF.md`.

## Recovered original delivery

The original project submission is now explicitly preserved in `archive/original_delivery/`:

- `Modelo Machine Learning.ipynb`: original notebook restored with its original delivery filename
- `README_original_2024.md`: earliest repository README recovered from history
- `original_theoretical_notes_es.md`: extracted theoretical and methodological explanations from the Spanish notebook
- `recovery_manifest.md`: provenance note explaining what was recovered and from which commits

## Executive Summary

- Dataset: `92,857` survey records and `14` raw variables.
- Valid analysis sample: `89,935` records.
- Trimmed modeling sample used for ML and econometrics: `86,313` records.
- Best predictive benchmark: Tuned Random Forest (`MAE = 880.05`, `RMSE = 1249.11`, `R^2 = 0.645`).
- Main tuned hyperparameters:
  - `n_estimators = 400`
  - `max_depth = 16`
  - `min_samples_leaf = 3`
- Econometric layer:
  - each education step is associated with a `12.47%` income premium in the OLS semilog model;
  - the estimated education premium rises from `11.46%` at the 25th percentile to `14.17%` at the 75th percentile.

## Why it works in a portfolio

This repository now demonstrates three complementary strengths:
- the ability to build and tune predictive models honestly;
- the ability to ask a more econometric question about heterogeneous returns;
- the ability to preserve the original academic framing, commercial context, and hypotheses instead of reducing the project to metrics alone.
