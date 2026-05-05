# Runbook

## 1. Validate Environment

```bash
scripts/validate_environment.sh
scripts/run_preflight.sh
```

Use `scripts/run_preflight.sh --require-model` before production scoring or monitoring when the approved champion reference should already exist.

## 2. Train

```bash
scripts/run_training.sh
```

Training behavior:

- Reads only `pai_rec_prod.alg_uplift_phase1_response_dataset_di`
- Uses the runner's current date for mature-label cutoff when `as_of_date` is omitted from config
- Uses time-based non-overlapping train, validation, and test partitions
- Excludes immature labels using `response_window_days + maturity_buffer_days`
- Selects the threshold with the precision-first policy plus recall floor
- Writes the champion reference to `contracts/model_registry/response_current.json`
- Writes runtime artifacts under `runtime/`, not inside the handoff package payload

## 3. Score

```bash
scripts/run_scoring.sh <YYYYMMDD>
```

Scoring behavior:

- Resolves the model from `contracts/model_registry/response_current.json`
- Uses the champion reference as the explicit approved model reference; no MLflow backend lookup is implemented here
- Reads the latest eligible population for the requested `pt`
- Writes only to `pai_rec_prod.alg_uplift_phase1_response_scores_di`
- Validates uniqueness on `player_id` within the scored `pt`

## 4. Monitor

```bash
scripts/run_monitoring.sh <YYYYMMDD> [REFERENCE_PT]
```

Monitoring behavior:

- Daily monitoring uses the same `pt` as scoring
- If `REFERENCE_PT` is omitted, the runtime resolves the previous available scored business partition
- Performance monitoring computes scalar delayed metrics only when labels are mature
- Immature labels write `status=not_evaluable`
- Missing score partitions emit an alert row and fail the run
