
# Response Model VIP-Level Reassessment

This reassessment uses the saved LightGBM prediction artifacts in `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm`. No business target definition was changed. The target remains `response_label_positive_3d`.

## Test VIP Metrics For Business Segments

| segment_value | row_count | positive_count | prevalence | pr_auc | roc_auc | precision | recall | f1 | predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V3 | 558380 | 313318 | 0.5611196676098714 | 0.7602996532622808 | 0.7178505237698102 | 0.6977858672644284 | 0.7133487383425147 | 0.7054814842872114 | 0.5736344424943587 |
| V1 | 440892 | 171507 | 0.3890000272175499 | 0.566282907526728 | 0.6723333862406976 | 0.5993190343749072 | 0.3530526450815419 | 0.4443457841050855 | 0.22915589305317402 |
| V4 | 232768 | 144587 | 0.6211635620016497 | 0.8175167130860623 | 0.7340366323974108 | 0.7286397201958119 | 0.7895868923208864 | 0.7578899849967471 | 0.6731208757217487 |
| V2 | 164784 | 83699 | 0.5079315953005146 | 0.6865825823793101 | 0.6882592551338632 | 0.6445448268598378 | 0.6580365356814298 | 0.651220809932013 | 0.5185636955044179 |
| V5 | 23978 | 15902 | 0.6631912586537659 | 0.8397751609465112 | 0.7286499002153419 | 0.7509571876087713 | 0.8140485473525343 | 0.7812311406155703 | 0.7189089999165902 |
| V6 | 2504 | 1617 | 0.6457667731629393 | 0.8325163271829434 | 0.7448564749257292 | 0.7620446533490012 | 0.8021026592455164 | 0.7815607110575474 | 0.6797124600638977 |
| V7 | 741 | 480 | 0.6477732793522267 | 0.8409954279565217 | 0.7521950830140487 | 0.7770700636942676 | 0.7625 | 0.7697160883280757 | 0.6356275303643725 |
| V8 | 663 | 443 | 0.6681749622926093 | 0.8662779230643392 | 0.766791504206854 | 0.8013856812933026 | 0.7832957110609481 | 0.7922374429223744 | 0.6530920060331825 |

## V1 And V2 Focus

| segment | split | row_count | positive_count | prevalence | pr_auc | roc_auc | selected_threshold | selected_f1 | best_threshold_on_grid | best_f1_on_grid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | validation | 193980 | 81483 | 0.4201 | 0.6057 | 0.6834 | 0.50 | 0.4457 | 0.30 | 0.6230 |
| V1 | test | 440892 | 171507 | 0.3890 | 0.5663 | 0.6723 | 0.50 | 0.4443 | 0.30 | 0.5872 |
| V2 | validation | 65958 | 35107 | 0.5323 | 0.7000 | 0.6832 | 0.50 | 0.6620 | 0.35 | 0.7067 |
| V2 | test | 164784 | 83699 | 0.5079 | 0.6866 | 0.6883 | 0.50 | 0.6512 | 0.35 | 0.6846 |

## Commentary

- `V1` remains materially weak. Test PR-AUC is `0.5663` and ROC-AUC is `0.6723` on `440,892` rows, so there is some ranking signal but not enough separation for a strong operating point.
- `V1` is underperforming for both reasons. The current global threshold `0.50` is too conservative for this segment: test recall is only `0.3531` with a predicted-positive rate of `0.2292` against prevalence `0.3890`. Lowering the threshold to `0.30` improves test F1 from `0.4443` to `0.5872`, but the segment still only reaches precision `0.4503` and recall `0.8436`. That means threshold tuning helps, but weak ranking remains a real limitation.
- `V2` is better than `V1`, but still clearly below `V3`, `V4`, and `V5`. Test PR-AUC is `0.6866` and ROC-AUC is `0.6883` on `164,784` rows.
- `V2` underperformance is mostly ranking-related rather than threshold-related. The current threshold `0.50` gives test F1 `0.6512`, while the best grid threshold `0.35` only nudges F1 to `0.6846`. Threshold changes shift recall and volume, but do not fix the segment’s moderate ranking quality.
- `V1` and `V2` are both large enough to interpret. Their test row counts are `440,892` and `164,784` respectively, so their weak performance should be treated as real rather than as a pure small-sample artifact.
- `V6`, `V7`, and `V8` are too small for strong threshold conclusions in the current saved test slice. They have only `V7=741 rows, V8=663 rows`.

## Segments Too Small To Interpret Strongly

Threshold-grid sufficiency was defined explicitly as `row_count >= 1000` and `positive_count >= 100` within a split. The following test segments did not meet that bar:

| segment_value | row_count | positive_count | prevalence |
| --- | --- | --- | --- |
| V7 | 741 | 480 | 0.6477732793522267 |
| V8 | 663 | 443 | 0.6681749622926093 |
| UNKNOWN | 123 | 23 | 0.18699186991869918 |

## Practical Next Recommendations

- Keep `V1` explicitly flagged as a weak segment in business-facing evaluation. Do not represent it as production-ready classification quality at the saved threshold.
- Keep `V2` visible as a middling segment: usable as part of a global ranking view, but materially weaker than `V3` to `V5`.
- Use the threshold grid when discussing operations. For `V1`, threshold choice changes the tradeoff materially. For `V2`, threshold choice mostly changes volume and recall, not underlying ranking quality.
- If the score is used operationally, keep one documented global threshold first and treat any later VIP-aware thresholding as a separate policy decision rather than a silent model-quality fix.
