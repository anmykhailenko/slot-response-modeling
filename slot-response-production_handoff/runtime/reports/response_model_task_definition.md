
# Phase-1 Response Model Task Definition

## Recommended Modeling Task

- Recommended task: binary response modeling.
- Exact target column: `response_label_positive_3d`.

## Why This Is The Best Phase-1 Choice

- `response_label_positive_3d` is the cleanest directly observed label already present in the audited table.
- It avoids the heavy zero-inflation and long-tail instability of raw post-treatment value regression targets.
- There is no audited multiclass response label in the live schema, so multiclass modeling would require inventing bins or semantics that the table does not define.
- A binary response baseline is the most defensible first step for an observational dataset where all rows are already treated.

## What This Answers For The Business

- It estimates which historically exposed users look most likely to show a positive 3-day betting response after a marketing exposure.
- It supports ranking or prioritizing likely responders within future marketing candidate pools, provided the business treats the score as observational propensity to react, not proof of lift.

## What It Does Not Answer

- It does not estimate incremental causal effect.
- It does not tell the business whether a user would have responded without treatment.
- It does not prove that SMS or vouchers cause the observed response.
- It does not optimize treatment-vs-no-treatment policy because the dataset has no untreated control rows.
