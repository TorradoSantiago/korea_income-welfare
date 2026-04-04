# Korea Income & Welfare

## Executive summary

This project studies how education relates to income using Korean welfare microdata and now includes a stronger English analysis pipeline outside the notebook. The updated workflow processes `89,935` valid records, exports reusable figures, and compares an education-only baseline against richer multivariable models.

## Analytical question

How much of individual income can education explain, and how much explanatory power appears only after bringing age, household structure, region, and social attributes into the model?

## Updated hypotheses

- `H1`: education has a meaningful positive relationship with income.
- `H2`: education is informative but incomplete; a broader socioeconomic model should materially outperform an education-only benchmark.

## What the upgraded analysis shows

- Income rises steadily across the education ladder, from roughly `1,004 USD` on average for people with no formal education to `8,247 USD` for doctorate holders.
- Seoul leads the regional distribution with a median income of roughly `2,884 USD`.
- The education-only linear regression reaches `R² = 0.294`, which confirms signal but leaves most variance unexplained.
- The multivariable linear model improves to `R² = 0.545`.
- The multivariable random forest reaches `R² = 0.628`, which is a substantial methodological improvement over the original one-feature benchmark.

## Most important modeling lesson

The strongest feature in the random-forest model is not education. It is `family_member`, followed by education, age, and year. That changes the interpretation in a useful way: the original hypothesis was directionally right, but the improved model shows that household structure and lifecycle effects are central to the income story.

## Improved conclusion

Education matters and should stay at the center of the narrative, but the portfolio-quality conclusion is now more mature: education is a strong socioeconomic signal, not a sufficient explanation by itself. Once the model includes age, year, household size, and demographic context, predictive performance improves sharply. That makes the repository more credible for analytics, policy, and applied economics positioning because it shows both signal detection and methodological restraint.

## Portfolio takeaway

This repository now demonstrates:

- a reproducible analysis pipeline in English;
- reusable charts and benchmark tables;
- a more senior interpretation of what the model can and cannot explain;
- a better bridge between exploratory machine learning and socioeconomic storytelling.
