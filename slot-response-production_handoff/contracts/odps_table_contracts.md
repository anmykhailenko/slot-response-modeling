# ODPS Table Contracts

All production source and sink tables used by this package are real ODPS tables. Do not rename them in configs without updating runtime code, DDL, validation SQL, and downstream schedulers together.

## Source Tables

| Role | Table |
| --- | --- |
| Training dataset | `pai_rec_prod.alg_uplift_phase1_response_dataset_di` |
| Scoring feature population | `pai_rec_prod.ads_bp_user_slot_churn_features_v2_di` |
| Delayed outcome source | `SuperEngineProject.ads_bet_site_order_sum_di` |

## Sink Tables

| Role | Table |
| --- | --- |
| Scoring output | `pai_rec_prod.alg_uplift_phase1_response_scores_di` |
| Monitoring daily | `pai_rec_prod.ads_uplift_phase1_response_monitoring_daily_di` |
| Monitoring performance | `pai_rec_prod.ads_uplift_phase1_response_monitoring_performance_di` |
| Monitoring alerts | `pai_rec_prod.ads_uplift_phase1_response_monitoring_alerts_di` |

## Non-Negotiable Rules

- Scoring must never write to the retired response predictions sink.
- Training must read only `pai_rec_prod.alg_uplift_phase1_response_dataset_di`.
- Monitoring must read scores from `pai_rec_prod.alg_uplift_phase1_response_scores_di`.
- The package must not depend on local CSV, Parquet, pickle, runtime dump directories, or experiment-tracking directories as production inputs.
