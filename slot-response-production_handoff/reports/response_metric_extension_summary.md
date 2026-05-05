
# Response Metric Extension Summary

The response-model evaluation outputs now include the additional thresholded classification metrics requested for deployment review.

## Added Metrics

- `positive_prediction_count`
- `negative_prediction_count`
- `predicted_positive_rate`
- `predicted_negative_rate`
- `specificity`
- `negative_predictive_value`

## Where They Now Appear

- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/metrics.csv` for global train, validation, and test metrics at the selected threshold
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/segment_metrics.csv` for thresholded segment metrics
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/vip_level_metrics_validation.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/vip_level_metrics_test.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/threshold_table.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/threshold_grid_global.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/threshold_grid_vip_level.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/threshold_grid_vip_level_technical.csv`

## Global Selected-Threshold Metrics

| threshold | precision | recall | f1 | specificity | negative_predictive_value | positive_prediction_count | negative_prediction_count | predicted_positive_rate | predicted_negative_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.6852652144325824 | 0.640059269303531 | 0.661891269362799 | 0.689778826611199 | 0.6448852219366219 | 683315.0 | 741518.0 | 0.47957550112890424 | 0.5204244988710958 |

## VIP-Level Support

- Global metrics are emitted for train, validation, and test.
- `vip_level` metrics are emitted where VIP reporting is already supported.
- Threshold-grid VIP outputs still respect the existing sample-size sufficiency rule before showing a segment.

## MLflow

- No active MLflow logging path is implemented in this response pipeline codebase today, so the extension was applied to persisted CSV reporting and markdown outputs only.
