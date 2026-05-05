# Schedule

## Daily Scoring

- `pt` must be the latest available ODPS business partition, usually Manila `T-1`
- Do not use `T-3` for daily scoring
- Run scoring after the eligible feature population partition is available

## Daily Monitoring

- Use the same `pt` as scoring
- `reference_pt` is the previous available scored business partition
- Daily monitoring compares the current scored partition to the previous available business partition when present

## Delayed Performance Monitoring

- Maturity horizon applies only to delayed performance evaluation
- Current runtime configuration uses `response_window_days=3` and `label_maturity_days=4`
- A partition is evaluable only when labels are mature as of the monitoring run date
- Immature partitions must write `status=not_evaluable`

## Suggested Order

1. Validate environment and configs
2. Run scoring for the latest available business `pt`
3. Run monitoring for the same `pt`
4. Run delayed performance monitoring as part of the same monitoring command once maturity conditions are satisfied
