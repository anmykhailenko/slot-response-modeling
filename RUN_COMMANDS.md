# Response Run Commands

Run from `response_modeling/`:

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
python3 src/monitoring/run_response_monitoring.py --config configs/response_monitoring.yaml --mode daily --pt 20260420 --reference-pt 20260419
python3 src/monitoring/run_response_monitoring.py --config configs/response_monitoring.yaml --mode performance --pt 20260420
```
