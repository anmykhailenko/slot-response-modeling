# Slot Response Modeling Production Handoff v1

This package is the clean production handoff for Phase-1 Slot Response Modeling. It keeps the working training, scoring, and monitoring logic, preserves the real ODPS source and sink tables, and removes stale local artifacts from the delivery package.

## Package Scope

- Response model training
- Response scoring
- Response monitoring
- Delayed performance monitoring
- ODPS source and sink contracts
- Threshold policy
- Model registry contract
- Scheduling instructions
- Validation and preflight

## Canonical Commands

```bash
python src/modeling/run_phase1_response_modeling.py --config configs/response_model.yaml
python src/modeling/run_phase1_response_scoring.py --config configs/response_scoring.yaml --pt <YYYYMMDD>
python src/monitoring/run_response_monitoring.py --config configs/response_monitoring.yaml --pt <YYYYMMDD> --reference-pt <YYYYMMDD>
python checks/preflight_check.py --config configs/response_model.yaml
scripts/validate_environment.sh
```

## Real ODPS Table Map

- Training dataset: `pai_rec_prod.alg_uplift_phase1_response_dataset_di`
- Scoring output: `pai_rec_prod.alg_uplift_phase1_response_scores_di`
- Monitoring daily: `pai_rec_prod.ads_uplift_phase1_response_monitoring_daily_di`
- Monitoring performance: `pai_rec_prod.ads_uplift_phase1_response_monitoring_performance_di`
- Monitoring alerts: `pai_rec_prod.ads_uplift_phase1_response_monitoring_alerts_di`

Additional live sources used by the working runtime:

- Scoring feature population: `pai_rec_prod.ads_bp_user_slot_churn_features_v2_di`
- Delayed outcome source: `SuperEngineProject.ads_bet_site_order_sum_di`

## Handoff Rules

- Do not use local artifacts as the production source of truth.
- Do not change the ODPS sink table names without coordinated downstream changes.
- Do not reintroduce the retired response predictions sink.
- Do not ship local runtime dumps, experiment stores, notebooks, archives, or prior handoff bundles inside this package.
