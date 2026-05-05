
# Response Model Threshold Grid

This report uses the saved validation and test prediction parquets in `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm` and evaluates the explicit threshold grid `0.05, 0.10, ..., 0.95`.

## Output Files

- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/threshold_grid_global.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/threshold_grid_vip_level.csv`
- `/Users/anastasiia.m/Documents/My projects/slot-response-modeling-clean/slot-response-production_handoff/runtime/response_modeling/models/lightgbm/threshold_grid_vip_level_technical.csv`

## Threshold Grid Sufficiency Rule For VIP Segments

- A VIP segment is included in the threshold-grid VIP outputs only when `row_count >= 1000` and `positive_count >= 100` in that split.
- This keeps the multi-threshold view readable and avoids over-interpreting tiny VIP segments.

## Global Test Threshold Snapshot

| threshold | precision | recall | f1 | specificity | negative_predictive_value | positive_prediction_count | negative_prediction_count | predicted_positive_rate | predicted_negative_rate | row_count | positive_count | prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 0.5510629980933843 | 0.9442272573184467 | 0.6959567902167441 | 0.18824332101947763 | 0.761813861987239 | 1253530.0 | 171303.0 | 0.8797732786930117 | 0.1202267213069883 | 1424833.0 | 731576.0 | 0.5134468390330656 |
| 0.5 | 0.6852652144325824 | 0.640059269303531 | 0.661891269362799 | 0.689778826611199 | 0.6448852219366219 | 683315.0 | 741518.0 | 0.47957550112890424 | 0.5204244988710958 | 1424833.0 | 731576.0 | 0.5134468390330656 |
| 0.6 | 0.7502245910503967 | 0.4760093278073638 | 0.5824564103121631 | 0.8327604337208279 | 0.6009612181675855 | 464177.0 | 960656.0 | 0.3257764243248156 | 0.6742235756751844 | 1424833.0 | 731576.0 | 0.5134468390330656 |

## Test Threshold Snapshot For V1

| threshold | precision | recall | f1 | specificity | negative_predictive_value | positive_prediction_count | negative_prediction_count | predicted_positive_rate | predicted_negative_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 0.4276577082118866 | 0.9153795471904936 | 0.5829609234156874 | 0.22004565955788183 | 0.8033202330939152 | 367102.0 | 73790.0 | 0.8326347495531785 | 0.1673652504468215 |
| 0.3 | 0.45029864078757753 | 0.8435574058201706 | 0.5871638568494875 | 0.34438443120441004 | 0.7756661622200112 | 321289.0 | 119603.0 | 0.7287249485134681 | 0.27127505148653186 |
| 0.35 | 0.47692023018707325 | 0.7514037327922476 | 0.5834936906017812 | 0.4753085732316202 | 0.7501948124234665 | 270215.0 | 170677.0 | 0.6128825199822179 | 0.3871174800177821 |
| 0.45 | 0.5478217936058069 | 0.4981429329415126 | 0.5218025963397168 | 0.7382222469699501 | 0.6979272683882107 | 155954.0 | 284938.0 | 0.35372381444888995 | 0.6462761855511101 |
| 0.6 | 0.6912466578375271 | 0.1612937081285312 | 0.26155649896466626 | 0.9541325612042244 | 0.6411731396227733 | 40019.0 | 400873.0 | 0.09076826070783775 | 0.9092317392921623 |

## Test Threshold Snapshot For V2

| threshold | precision | recall | f1 | specificity | negative_predictive_value | positive_prediction_count | negative_prediction_count | predicted_positive_rate | predicted_negative_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 0.5147131921662194 | 0.9897370338952676 | 0.6772317213245423 | 0.03676388974532897 | 0.7763020833333333 | 160944.0 | 3840.0 | 0.9766967666763764 | 0.023303233323623607 |
| 0.3 | 0.5284929129302863 | 0.9604415823367065 | 0.6818118206838644 | 0.1154960843559228 | 0.7387977279899022 | 152108.0 | 12676.0 | 0.9230750558306632 | 0.07692494416933682 |
| 0.35 | 0.550763342114072 | 0.9042760367507378 | 0.684575413461408 | 0.23863846580748596 | 0.7071851472845552 | 137422.0 | 27362.0 | 0.8339523254684921 | 0.16604767453150793 |
| 0.45 | 0.6128010634970285 | 0.7490173120347914 | 0.6740966543550373 | 0.5114756120120861 | 0.663780409731114 | 102304.0 | 62480.0 | 0.6208369744635401 | 0.37916302553645986 |
| 0.6 | 0.7027484379648914 | 0.45151077073800167 | 0.5497872340425531 | 0.8028611950422396 | 0.5864442202363793 | 53776.0 | 111008.0 | 0.3263423633362462 | 0.6736576366637538 |

## Notes

- The current recommended threshold is `0.50` under policy `max_precision_with_min_recall(minimum_recall=0.60)`.
- `V1` shows a visibly different threshold preference from the global choice: F1 peaks lower, around `0.30` on test.
- `V2` is comparatively stable across `0.30` to `0.45`, which reinforces that its remaining gap is not mainly a threshold-selection problem.
