# Response Run Commands

Run from the project root:

Training:

```bash
python3 src/modeling/run_phase1_response_modeling.py --config-path configs/response_model.yaml
```

Scoring:

```bash
python3 src/modeling/run_phase1_response_scoring.py --config-path configs/response_scoring.yaml --scoring-pt 20260420
```

Monitoring:

```bash
python3 src/monitoring/run_response_monitoring.py --config configs/response_monitoring.yaml --dry-run --assignment_start_date 20260420 --assignment_end_date 20260420
python3 src/monitoring/run_response_monitoring.py --config configs/response_monitoring.yaml --assignment_start_date 20260420 --assignment_end_date 20260420
```
