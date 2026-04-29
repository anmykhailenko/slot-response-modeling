# Response Dataset Rebuild SQL Package

Date: `2026-04-29`

## Goal

Repair `pai_rec_prod.alg_uplift_phase1_response_dataset_di` so that:

- `outcome_gross_bet_3d_value`
- `outcome_gross_ggr_3d_value`
- `outcome_source_row_count_3d`
- `outcome_distinct_source_keys_3d`
- `first_outcome_event_date`
- `last_outcome_event_date`
- `response_label_positive_3d`

are recomputed from `SuperEngineProject.ads_bet_site_order_sum_di` using the confirmed live logic:

- exact key: `cast(player_id as string) = cast(login_name as string)`
- window: `stat_date > assignment_date` and `stat_date <= assignment_date + 3 days`
- partition pruning: `o.pt` aligned to `stat_date` in `yyyymmdd`

## Confirmed Findings Behind This Package

- the current response dataset stores zero for every outcome and label row
- exact `player_id = login_name` matches do exist in the outcome source
- `stat_date` and `pt` are aligned in the outcome source
- many rows should be positive in the 3-day post-assignment window
- the current builder behavior is therefore wrong in the post-assignment outcome join / label construction path

## Runtime Parameters

- `${assignment_start_date}`: lower assignment partition bound in `yyyymmdd`
- `${assignment_end_date}`: upper assignment partition bound in `yyyymmdd`

## Safe Execution Pattern

1. Run `insert_overwrite_alg_uplift_phase1_response_dataset_di.sql`.
2. Run `validate_response_dataset_rebuild_concrete.sql`.

## Important Constraint

This package is concrete and uses only the real ODPS tables confirmed in live inspection:

- `pai_rec_prod.ads_bp_user_slot_churn_features_v2_di`
- `SuperEngineProject.dwd_mms_user_claimed_info_di`
- `superengineproject.ods_msg_sms_send_record`
- `SuperEngineProject.ads_bet_site_order_sum_di`
- `pai_rec_prod.alg_uplift_phase1_response_dataset_di`
