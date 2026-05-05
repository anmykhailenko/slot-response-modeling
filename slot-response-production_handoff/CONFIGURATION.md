# Configuration

## Environment Variables

- `ALIBABA_CLOUD_ACCESS_KEY_ID`
- `ALIBABA_CLOUD_ACCESS_KEY_SECRET`
- `ODPS_PROJECT`
- `ODPS_ENDPOINT`

## Core Config Files

- `configs/response_model.yaml`
- `configs/response_scoring.yaml`
- `configs/response_monitoring.yaml`

## Important Config Values

### Training

- `source_table: pai_rec_prod.alg_uplift_phase1_response_dataset_di`
- `target_column: response_label_positive_3d`
- `champion_reference_path: contracts/model_registry/response_current.json`
- `threshold_selection.policy: max_precision_with_min_recall`
- `threshold_selection.minimum_recall: 0.60`

### Scoring

- `publish.target_table: pai_rec_prod.alg_uplift_phase1_response_scores_di`
- `publish.partition_column: pt`
- `publish.write_mode: overwrite`

### Monitoring

- `response_monitoring.source.scored_table: pai_rec_prod.alg_uplift_phase1_response_scores_di`
- `response_monitoring.outputs.odps_daily_table: pai_rec_prod.ads_uplift_phase1_response_monitoring_daily_di`
- `response_monitoring.outputs.odps_performance_table: pai_rec_prod.ads_uplift_phase1_response_monitoring_performance_di`
- `response_monitoring.outputs.odps_alerts_table: pai_rec_prod.ads_uplift_phase1_response_monitoring_alerts_di`

## Model Registry

- Training creates or refreshes `contracts/model_registry/response_current.json`
- Scoring and monitoring expect that champion reference to point to an approved production bundle
- Do not replace it with a dummy file
- There is no active MLflow backend integration in this package. `mlflow_run_id` is stored for lineage only.
- The production source of truth is the approved champion reference JSON plus its referenced bundle.

## Date Handling

- If `as_of_date` is omitted from `configs/response_model.yaml`, training uses the runner's current date to compute the mature-label cutoff.
- If an explicit `as_of_date` is provided in training config, it must not be a future date.
- Scoring resolves the actual scoring partition from `--pt` or the latest available ODPS partition.
