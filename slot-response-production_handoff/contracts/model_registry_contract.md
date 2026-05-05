# Model Registry Contract

The production scoring runtime resolves the approved model from:

`contracts/model_registry/response_current.json`

This file is intentionally not shipped with fake content. It is created by training or supplied manually from an approved model bundle.

## Required Fields

- `model_dir`
- `export_bundle_path`
- `model_name`
- `model_version`
- `mlflow_run_id`
- `iteration_id`
- `threshold_artifact_path`
- `feature_schema_path`
- `run_metadata_path`
- `selected_threshold`
- `selected_score_variant`

## Production Rules

- The reference must point to a real approved bundle, not a local dev bundle.
- `selected_threshold` is the production threshold source of truth for scoring and delayed performance monitoring.
- The scoring runtime reads the model bundle from the resolved champion reference, not from an ad hoc local path argument.
- `model_version` and `mlflow_run_id` are persisted into scored output for lineage.
