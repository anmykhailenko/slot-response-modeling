
# Response Model Business Threshold Report

## What The Model Predicts

- The model predicts the probability of a positive observed 3-day response after historical treatment patterns similar to the training data.
- `label = 1` means a positive response was observed.
- `label = 0` means no positive response was observed in the response window. It does not mean harm.
- The prediction score is the estimated probability of positive response, not proof of causal uplift.

## Recommended Operating Point

- Recommended threshold: `0.50`
- Users predicted positive on held-out test data: `683,315` (`0.4796`)
- Users predicted negative on held-out test data: `741,518` (`0.5204`)

## Operational Meaning

- Users above the threshold should be considered for treatment within already-approved campaign pools because they are more likely to show a positive observed response.
- Users below the threshold should generally not be touched for broad treatment pushes and can be deprioritized when budget or contact capacity is limited.

## Precision And Recall Trade-Off

- At `0.35`, the old operating point delivered test precision `0.5783` and recall `0.8862`.
- At `0.50`, the recommended operating point improves test precision to `0.6853` while recall moves to `0.6401`.
- At `0.60`, precision rises further to `0.7502`, but recall drops to `0.4760`, which is likely too restrictive for default use.

## Caveat

- This is observational response modeling. It helps rank likely responders among already-eligible users, but it does not prove incremental uplift or causal treatment effect.
