# Response Scoring Output SQL Package

Date: `2026-04-21`

## Package Goal

Publish the Phase-1 response scoring output directly into the governed final ODPS table:

- `pai_rec_prod.alg_uplift_phase1_response_scores_di`

This package is designed for direct partition overwrite of the final table. It does not support append writes.

## Direct-Write Contract

The normal production path is:

1. run `response_modeling/src/modeling/run_phase1_response_scoring.py`
2. expose the scored rows to ODPS as `${direct_scored_source}` for the current publish run
3. run `validate_response_scores_prewrite_contract.sql`
4. if and only if every pre-write violation count is zero, run `insert_overwrite_alg_uplift_phase1_response_scores_di.sql`
5. run `validate_response_scores_postwrite_contract.sql`

`${direct_scored_source}` should be a run-scoped ODPS relation such as a temporary view, transient external table, or equivalent execution-time binding. A persistent `staging_scored_table` is not part of the target design.

## Runtime Parameters

- `${direct_scored_source}`: scored source relation bound for the current run
- `${scoring_pt}`: partition to publish in `yyyymmdd` form

## Production Guarantees

- direct write to final table only
- overwrite only `pt = '${scoring_pt}'`
- no append behavior
- one row per `player_id` in the target partition, enforced by mandatory pre-write uniqueness validation
- required metadata must be present before publish

## Remaining Technical Boundary

Model inference still happens in Python, not SQL. Because of that, some execution environments may still need a transient adapter to make the Python-scored dataset visible inside ODPS for the publish step.

That adapter is only technically required when the runtime cannot bind the scored output directly into the same ODPS session. If an adapter is needed, use a temporary or transient run-scoped relation, not a durable shared staging table.
