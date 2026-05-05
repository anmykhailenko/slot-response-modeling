# Monitoring Status Contract

The delayed performance monitoring table uses only these `status` values:

- `completed`
- `not_evaluable`
- `failed`

Operational rule:

- Use `completed` when mature labels exist and scalar performance metrics are written.
- Use `not_evaluable` when labels are not mature yet.
- Use `failed` only for an explicit failed write contract if the runtime is extended to persist failure rows later.

The current runtime writes `completed` and `not_evaluable`. Missing score partitions emit an alert and fail the run instead of silently skipping writes.
