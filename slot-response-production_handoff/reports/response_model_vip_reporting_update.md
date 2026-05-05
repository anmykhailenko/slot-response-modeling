
# Response Model VIP Reporting Update

The VIP reporting split is now explicit. Business-facing files exclude `NULL` and `UNKNOWN`. Technical files preserve them for debugging and data-quality review.

## Business-Facing Outputs

- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/vip_level_metrics_validation.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/vip_level_metrics_test.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/threshold_grid_vip_level.csv`

These files only include named VIP business segments and exclude `UNKNOWN` plus physical nulls represented as `__NULL__`.

## Technical Outputs

- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/vip_level_metrics_validation_technical.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/vip_level_metrics_test_technical.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/threshold_grid_vip_level_technical.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/segment_metrics.csv`

These files keep all persisted categories, including `UNKNOWN` and `__NULL__`, because they are still useful for model debugging, score-distribution review, and upstream data-quality checks.

## Where NULL And UNKNOWN Still Remain

- `UNKNOWN` remains in the technical VIP files because it is a real modeled category in the saved predictions.
- `__NULL__` remains in the technical VIP files because the saved prediction artifacts contain physical null `vip_level` values that are still important to audit.
- `threshold_grid_global.csv` remains global only and therefore has no VIP category rows.

## Current Business/Test VIP View

| segment_value | row_count | positive_count | prevalence | pr_auc | roc_auc | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V3 | 558380 | 313318 | 0.5611196676098714 | 0.7602996532622808 | 0.7178505237698102 | 0.6977858672644284 | 0.7133487383425147 | 0.7054814842872114 |
| V1 | 440892 | 171507 | 0.3890000272175499 | 0.566282907526728 | 0.6723333862406976 | 0.5993190343749072 | 0.3530526450815419 | 0.4443457841050855 |
| V4 | 232768 | 144587 | 0.6211635620016497 | 0.8175167130860623 | 0.7340366323974108 | 0.7286397201958119 | 0.7895868923208864 | 0.7578899849967471 |
| V2 | 164784 | 83699 | 0.5079315953005146 | 0.6865825823793101 | 0.6882592551338632 | 0.6445448268598378 | 0.6580365356814298 | 0.651220809932013 |
| V5 | 23978 | 15902 | 0.6631912586537659 | 0.8397751609465112 | 0.7286499002153419 | 0.7509571876087713 | 0.8140485473525343 | 0.7812311406155703 |
| V6 | 2504 | 1617 | 0.6457667731629393 | 0.8325163271829434 | 0.7448564749257292 | 0.7620446533490012 | 0.8021026592455164 | 0.7815607110575474 |
| V7 | 741 | 480 | 0.6477732793522267 | 0.8409954279565217 | 0.7521950830140487 | 0.7770700636942676 | 0.7625 | 0.7697160883280757 |
| V8 | 663 | 443 | 0.6681749622926093 | 0.8662779230643392 | 0.766791504206854 | 0.8013856812933026 | 0.7832957110609481 | 0.7922374429223744 |
