
# Phase-1 Observational Response Dataset Audit

This audit is for `pai_rec_prod.alg_uplift_phase1_response_dataset_di`.

This table is treated as an observational response dataset. It is not a causal uplift dataset, and nothing in this audit should be read as proof of treatment effect.

## Live Table Coverage

- Partition coverage discovered from ODPS metadata: `20260201` to `20260503` (`74` daily partitions).
- Conservative modeling cutoff for a 3-day response label on `as_of_date=2026-05-05`: `pt <= 20260501`.
- Full table rows across all live partitions: `5,891,540`.
- Mature modeling rows through `20260501`: `5,891,540`.
- Distinct `(player_id, assignment_date)` keys across mature rows: `5,891,540`.
- Mature-table positive response rate: `0.5061`.

## Schema Summary

| column_name | data_type | comment | is_partition |
| --- | --- | --- | --- |
| player_id | STRING | login_name-style player identifier | False |
| assignment_date | DATE | treatment assignment date | False |
| treatment_timestamp | TIMESTAMP | earliest observed treatment timestamp on assignment_date | False |
| treatment_flag | BIGINT | binary treatment indicator; always 1 for this observational treated dataset | False |
| raw_assignment_event_count | BIGINT | count of raw treatment events collapsed into the user-day row | False |
| has_voucher_treatment | BIGINT | 1 if voucher source contributed to the user-day row | False |
| has_sms_treatment | BIGINT | 1 if sms source contributed to the user-day row | False |
| feature_snapshot_date | DATE | selected pre-treatment feature snapshot date | False |
| recent_bet_cnt_7d | BIGINT | feature mart column | False |
| recent_bet_amt_7d | DOUBLE | feature mart column | False |
| recent_win_amt_7d | DOUBLE | feature mart column | False |
| recent_ggr_amt_7d | DOUBLE | feature mart column | False |
| recent_net_loss_amt_7d | DOUBLE | feature mart column | False |
| recent_bet_days_7d | BIGINT | feature mart column | False |
| recency_last_bet_to_t | BIGINT | feature mart column | False |
| pre_bet_cnt_30d | BIGINT | feature mart column | False |
| pre_bet_amt_30d | DOUBLE | feature mart column | False |
| pre_win_amt_30d | DOUBLE | feature mart column | False |
| pre_ggr_amt_30d | DOUBLE | feature mart column | False |
| pre_net_loss_amt_30d | DOUBLE | feature mart column | False |
| pre_bet_days_30d | BIGINT | feature mart column | False |
| recent_dep_cnt_7d | BIGINT | feature mart column | False |
| recent_dep_amt_7d | DOUBLE | feature mart column | False |
| recent_withdraw_cnt_7d | BIGINT | feature mart column | False |
| recent_withdraw_amt_7d | DOUBLE | feature mart column | False |
| recent_net_cash_in_7d | DOUBLE | feature mart column | False |
| feature_create_time | TIMESTAMP | raw feature mart audit column | False |
| data_version | STRING | raw feature mart audit column | False |
| vip_level | STRING | raw feature mart audit column | False |
| outcome_gross_bet_3d_value | DOUBLE | summed post-treatment 3-day bet amount from the isolated outcome source | False |
| outcome_gross_ggr_3d_value | DOUBLE | summed post-treatment 3-day gross gaming revenue from the isolated outcome source | False |
| outcome_source_row_count_3d | BIGINT | count of source rows contributing to the 3-day outcome rollup | False |
| outcome_distinct_source_keys_3d | BIGINT | count of distinct outcome source keys contributing to the 3-day rollup | False |
| first_outcome_event_date | DATE | earliest post-treatment outcome date observed in the 3-day window | False |
| last_outcome_event_date | DATE | latest post-treatment outcome date observed in the 3-day window | False |
| response_label_positive_3d | BIGINT | 1 if outcome_gross_bet_3d_value > 0 else 0 | False |
| pt | STRING | assignment_date partition in yyyymmdd form | True |

## Partition Summary

| pt | row_count | positive_rate | mature_for_modeling |
| --- | --- | --- | --- |
| 20260217 | 60824 | 0.5164737603577535 | True |
| 20260218 | 62294 | 0.5354448261469804 | True |
| 20260219 | 63677 | 0.5292491794525496 | True |
| 20260220 | 65258 | 0.5176376842685955 | True |
| 20260221 | 65717 | 0.48466911149322095 | True |
| 20260222 | 63920 | 0.47648623279098873 | True |
| 20260223 | 60456 | 0.4868995633187773 | True |
| 20260224 | 58345 | 0.48737681035221525 | True |
| 20260225 | 68035 | 0.43324759315058425 | True |
| 20260226 | 67753 | 0.4517438342213629 | True |
| 20260227 | 65063 | 0.473218265373561 | True |
| 20260228 | 64735 | 0.46966864910790146 | True |
| 20260301 | 63839 | 0.45033600150378295 | True |
| 20260302 | 61511 | 0.45419518460112823 | True |
| 20260303 | 59200 | 0.4729222972972973 | True |
| 20260304 | 53787 | 0.48727387658727944 | True |
| 20260305 | 51256 | 0.49018651474949276 | True |
| 20260306 | 53099 | 0.4807058513343001 | True |
| 20260307 | 54547 | 0.48994445157387206 | True |
| 20260308 | 54577 | 0.49009656082232445 | True |

## Modeling Grain

- Confirmed row grain: one treated `player_id` per `assignment_date`.
- Live ODPS check: `total_rows == distinct(player_id, assignment_date)` on the audited mature range.
- Treatment semantics: every row is already treated (`treatment_flag = 1`), so this is response modeling within exposed rows, not treat-vs-control uplift estimation.

## Candidate Target Column(s)

| target_column | task_family | status | notes |
| --- | --- | --- | --- |
| response_label_positive_3d | binary classification | recommended | Directly observed binary response flag derived from positive 3-day gross bet. |
| outcome_gross_bet_3d_value | regression | secondary only | Heavy zero inflation and long-tailed spend make it a weaker first baseline. |
| outcome_gross_ggr_3d_value | regression | secondary only | More volatile commercial proxy and harder to communicate as a first response model. |

## Candidate Feature Column Groups

| group | columns |
| --- | --- |
| Pre-treatment recent betting behavior | recent_bet_cnt_7d, recent_bet_amt_7d, recent_win_amt_7d, recent_ggr_amt_7d, recent_net_loss_amt_7d, recent_bet_days_7d, recency_last_bet_to_t |
| Pre-treatment 30-day betting behavior | pre_bet_cnt_30d, pre_bet_amt_30d, pre_win_amt_30d, pre_ggr_amt_30d, pre_net_loss_amt_30d, pre_bet_days_30d |
| Pre-treatment cashflow behavior | recent_dep_cnt_7d, recent_dep_amt_7d, recent_withdraw_cnt_7d, recent_withdraw_amt_7d, recent_net_cash_in_7d |
| Pre-treatment categorical segment | vip_level |
| Observed treatment metadata | has_voucher_treatment, has_sms_treatment, treatment_flag, raw_assignment_event_count |

## Columns To Exclude From Modeling

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

## Treatment Indicator Columns Present

- `treatment_flag`
- `has_voucher_treatment`
- `has_sms_treatment`
- `raw_assignment_event_count`

## Key Data Quality Caveats

- The table contains very recent partitions through `20260503`. For a 3-day post-treatment response label, partitions after `20260501` are conservatively treated as immature and excluded from training/evaluation.
- `vip_level` is missing for `766,077` mature rows, so the baseline keeps it with explicit missing-category handling rather than dropping those rows.
- Positive-rate drift is material across partitions, from roughly `0.4332` to `0.5466`. Time-based validation is therefore required.
- The dataset remains observational only. All rows are exposed rows, so model outputs estimate likely response conditional on historical exposure patterns, not incremental causal lift.

## Sample Rows

| player_id | assignment_date | treatment_timestamp | treatment_flag | raw_assignment_event_count | has_voucher_treatment | has_sms_treatment | feature_snapshot_date | recent_bet_cnt_7d | recent_bet_amt_7d | recent_win_amt_7d | recent_ggr_amt_7d | recent_net_loss_amt_7d | recent_bet_days_7d | recency_last_bet_to_t | pre_bet_cnt_30d | pre_bet_amt_30d | pre_win_amt_30d | pre_ggr_amt_30d | pre_net_loss_amt_30d | pre_bet_days_30d | recent_dep_cnt_7d | recent_dep_amt_7d | recent_withdraw_cnt_7d | recent_withdraw_amt_7d | recent_net_cash_in_7d | feature_create_time | data_version | vip_level | outcome_gross_bet_3d_value | outcome_gross_ggr_3d_value | outcome_source_row_count_3d | outcome_distinct_source_keys_3d | first_outcome_event_date | last_outcome_event_date | response_label_positive_3d | pt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afloro1983 | 2026-05-01 |  | 0 | 0 | 0 | 0 | 2026-05-01 | 79 | 79.0 | -19.7 | 98.7 | 98.7 | 1 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 | 20.0 | 0 | 0.0 | 20.0 | 2026-05-04 06:00:57.164000 | v20260503 | V3 | 0.0 | 0.0 | 0 | 0 |  |  | 0 | 20260501 |
| aidl58 | 2026-05-01 | 2026-05-01 10:32:47 | 1 | 2 | 0 | 1 | 2026-05-01 | 418 | 808.0 | -307.1 | 1115.1 | 1115.1 | 1 | 2 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 | 300.0 | 0 | 0.0 | 300.0 | 2026-05-04 06:00:57.164000 | v20260503 | V1 | 0.0 | 0.0 | 0 | 0 |  |  | 0 | 20260501 |
| al25sxjy | 2026-05-01 | 2026-05-01 01:52:50 | 1 | 2 | 0 | 1 | 2026-05-01 | 1793 | 5498.5 | -315.36 | 5813.86 | 5813.86 | 4 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 2 | 200.0 | 0 | 0.0 | 200.0 | 2026-05-04 06:00:57.164000 | v20260503 | V2 | 419.0 | 99.3 | 1 | 1 | 2026-05-03 | 2026-05-03 | 1 | 20260501 |
| al665u751q | 2026-05-01 | 2026-05-01 13:09:13 | 1 | 3 | 1 | 1 | 2026-05-01 | 1398 | 10184.5 | -687.3 | 10871.8 | 10871.8 | 4 | 2 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 32 | 3080.0 | 0 | 0.0 | 3080.0 | 2026-05-04 06:00:57.164000 | v20260503 | V3 | 1381.5 | 440.5 | 2 | 2 | 2026-05-02 | 2026-05-03 | 1 | 20260501 |
| al73wknu | 2026-05-01 | 2026-05-01 05:24:50 | 1 | 3 | 1 | 1 | 2026-05-01 | 4 | 2.0 | -2.0 | 4.0 | 4.0 | 1 | 2 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 7 | 400.0 | 0 | 0.0 | 400.0 | 2026-05-04 06:00:57.164000 | v20260503 | V2 | 4710.0 | -503.0 | 3 | 3 | 2026-05-02 | 2026-05-04 | 1 | 20260501 |

## Sampled Missingness Check

The detailed column-level missingness check below is computed on the reproducible local hash sample used for baseline training.

| column_name | missing_rows | missing_rate |
| --- | --- | --- |
| first_outcome_event_date | 2198161 | 0.4877742502906349 |
| last_outcome_event_date | 2198161 | 0.4877742502906349 |
| treatment_timestamp | 1072066 | 0.23789257902950686 |
| assignment_date | 0 | 0.0 |
| data_version | 0 | 0.0 |
| feature_create_time | 0 | 0.0 |
| feature_snapshot_date | 0 | 0.0 |
| has_sms_treatment | 0 | 0.0 |
| has_voucher_treatment | 0 | 0.0 |
| outcome_distinct_source_keys_3d | 0 | 0.0 |
| outcome_gross_bet_3d_value | 0 | 0.0 |
| outcome_gross_ggr_3d_value | 0 | 0.0 |
| outcome_source_row_count_3d | 0 | 0.0 |
| player_id | 0 | 0.0 |
| pre_bet_amt_30d | 0 | 0.0 |
| pre_bet_cnt_30d | 0 | 0.0 |
| pre_bet_days_30d | 0 | 0.0 |
| pre_ggr_amt_30d | 0 | 0.0 |
| pre_net_loss_amt_30d | 0 | 0.0 |
| pre_win_amt_30d | 0 | 0.0 |
| pt | 0 | 0.0 |
| raw_assignment_event_count | 0 | 0.0 |
| recency_last_bet_to_t | 0 | 0.0 |
| recent_bet_amt_7d | 0 | 0.0 |
| recent_bet_cnt_7d | 0 | 0.0 |
