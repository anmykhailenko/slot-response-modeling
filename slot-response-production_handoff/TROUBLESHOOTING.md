# Troubleshooting

## Missing ODPS Credentials

- Symptom: preflight or runtime fails before querying ODPS
- Fix: export `ALIBABA_CLOUD_ACCESS_KEY_ID`, `ALIBABA_CLOUD_ACCESS_KEY_SECRET`, `ODPS_PROJECT`, and `ODPS_ENDPOINT`

## Champion Reference Missing

- Symptom: scoring fails to resolve `contracts/model_registry/response_current.json`
- Fix: run training first or place the approved champion reference and bundle in the expected location, then rerun `scripts/run_preflight.sh --require-model`

## Future Partition Date

- Symptom: preflight rejects `pt` or `reference_pt`
- Fix: use a real available ODPS business partition in `YYYYMMDD` format

## Missing Score Partition During Monitoring

- Symptom: monitoring fails after writing an alert row
- Fix: confirm scoring completed successfully for the same `pt` and that the scored partition exists in `pai_rec_prod.alg_uplift_phase1_response_scores_di`

## Duplicate `player_id` Rows In Scoring

- Symptom: scoring post-write validation fails
- Fix: inspect the upstream eligible population query for unexpected duplicate players before re-running

## Labels Not Mature

- Symptom: performance monitoring writes `status=not_evaluable`
- Fix: wait until the maturity horizon has elapsed; this is expected behavior, not a silent failure
