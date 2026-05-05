
# Phase-1 Response Model Feature Selection

## Included Modeling Features

- Numeric pre-treatment features: `recent_bet_cnt_7d, recent_bet_amt_7d, recent_win_amt_7d, recent_ggr_amt_7d, recent_net_loss_amt_7d, recent_bet_days_7d, recency_last_bet_to_t, pre_bet_cnt_30d, pre_bet_amt_30d, pre_win_amt_30d, pre_ggr_amt_30d, pre_net_loss_amt_30d, pre_bet_days_30d, recent_dep_cnt_7d, recent_dep_amt_7d, recent_withdraw_cnt_7d, recent_withdraw_amt_7d, recent_net_cash_in_7d`
- Categorical pre-treatment features: `vip_level`

## Excluded Columns

| column_name | reason |
| --- | --- |
| player_id | Identifier; not a model feature. |
| assignment_date | Row key and temporal anchor, not a behavioral feature. |
| treatment_timestamp | Operational timestamp after assignment logic; excluded from the baseline. |
| treatment_flag | Constant 1 for all rows; no predictive value. |
| raw_assignment_event_count | Execution artifact rather than a stable pre-treatment covariate. |
| has_voucher_treatment | Observed channel metadata; excluded from the default baseline to avoid conditioning on post-decision execution. |
| has_sms_treatment | Observed channel metadata; excluded from the default baseline to avoid conditioning on post-decision execution. |
| feature_snapshot_date | Snapshot-control field rather than user behavior. |
| feature_create_time | Audit metadata. |
| data_version | Audit metadata. |
| outcome_gross_bet_3d_value | Post-treatment realized outcome and direct leakage for response_label_positive_3d. |
| outcome_gross_ggr_3d_value | Post-treatment realized outcome and leakage-risk field. |
| outcome_source_row_count_3d | Outcome window construction metadata after treatment. |
| outcome_distinct_source_keys_3d | Outcome window construction metadata after treatment. |
| first_outcome_event_date | Post-treatment event timing leakage. |
| last_outcome_event_date | Post-treatment event timing leakage. |
| response_label_positive_3d | Target column. |
| pt | Temporal split key, not a model feature. |

## Leakage Rationale

- All realized outcome columns and outcome-window metadata are excluded because they are observed after treatment and would leak the response label.
- Identifier, audit, and partition columns are excluded because they are not stable behavioral features.
- Treatment-channel fields (`has_voucher_treatment`, `has_sms_treatment`) are kept in the extracted dataset for auditing, but excluded from the default baseline because they describe realized execution rather than universally available pre-treatment state. They can be added later only for channel-conditional models where the planned channel is known at score time.

## Split Strategy

- Strategy: time-based split on `pt`.
- Sampling: disabled in the canonical runtime. Training uses the full configured mature ODPS split window.
- Mature modeling window used for model fitting/evaluation: `20260312` to `20260501`.
- Split manifest:

| split | distinct_pt_count | start_pt | end_pt |
| --- | --- | --- | --- |
| train | 30 | 20260312 | 20260410 |
| validation | 7 | 20260411 | 20260417 |
| test | 14 | 20260418 | 20260501 |
