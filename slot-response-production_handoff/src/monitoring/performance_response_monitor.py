from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from .config import ResponseMonitoringConfig


def mature_outcome_window_for_pt(pt: str, label_maturity_days: int, response_window_days: int) -> Dict[str, Any]:
    score_date = datetime.strptime(pt, "%Y%m%d").date()
    matured_on = score_date + timedelta(days=label_maturity_days)
    observed_end = score_date + timedelta(days=response_window_days)
    return {
        "scoring_pt": str(pt),
        "score_date": score_date.isoformat(),
        "observed_outcome_end_date": observed_end.isoformat(),
        "matured_on_date": matured_on.isoformat(),
        "label_maturity_days": int(label_maturity_days),
        "response_window_days": int(response_window_days),
    }


def labels_are_mature_for_pt(
    pt: str,
    *,
    as_of_date: date,
    label_maturity_days: int,
    response_window_days: int,
) -> Dict[str, Any]:
    summary = mature_outcome_window_for_pt(
        pt,
        label_maturity_days=label_maturity_days,
        response_window_days=response_window_days,
    )
    matured = as_of_date >= datetime.strptime(summary["matured_on_date"], "%Y-%m-%d").date()
    summary["as_of_date"] = as_of_date.isoformat()
    summary["labels_mature"] = bool(matured)
    return summary


def validate_observational_frame(frame: pd.DataFrame, *, response_window_days: int) -> None:
    required = {
        "player_id",
        "pt",
        "predicted_response_score",
        "selected_threshold",
        "model_name",
        "model_version",
        f"observed_response_label_positive_{response_window_days}d",
        f"observed_gross_bet_value_{response_window_days}d",
        f"observed_gross_ggr_value_{response_window_days}d",
    }
    missing = [column for column in sorted(required) if column not in frame.columns]
    if missing:
        raise ValueError(f"Delayed response monitoring input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Delayed response monitoring input is empty.")
    duplicate_count = int(frame.duplicated(subset=["player_id", "pt"]).sum())
    if duplicate_count:
        raise ValueError(f"Delayed response monitoring input contains duplicate player_id,pt rows: {duplicate_count}")


def _single_value_or_none(series: pd.Series) -> Any:
    values = series.dropna().astype(str).unique().tolist()
    if not values:
        return None
    if len(values) > 1:
        return ",".join(sorted(values))
    return values[0]


def _single_float_or_none(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[0])


def _confusion_counts(y_true: np.ndarray, predicted: np.ndarray) -> Dict[str, int]:
    return {
        "true_positive_count": int(((y_true == 1) & (predicted == 1)).sum()),
        "false_positive_count": int(((y_true == 0) & (predicted == 1)).sum()),
        "false_negative_count": int(((y_true == 1) & (predicted == 0)).sum()),
        "true_negative_count": int(((y_true == 0) & (predicted == 0)).sum()),
    }


def build_global_performance_row(frame: pd.DataFrame, *, pt: str, response_window_days: int) -> Dict[str, Any]:
    label_column = f"observed_response_label_positive_{response_window_days}d"
    bet_value_column = f"observed_gross_bet_value_{response_window_days}d"
    ggr_value_column = f"observed_gross_ggr_value_{response_window_days}d"

    y_true = pd.to_numeric(frame[label_column], errors="raise").astype(int).to_numpy()
    scores = pd.to_numeric(frame["predicted_response_score"], errors="raise").astype(float).to_numpy()
    threshold = _single_float_or_none(frame["selected_threshold"])
    if threshold is None:
        raise ValueError("Delayed response monitoring could not resolve a selected_threshold from the scored partition.")

    predicted = (scores >= float(threshold)).astype(int)
    confusion = _confusion_counts(y_true, predicted)
    bet_values = pd.to_numeric(frame[bet_value_column], errors="raise").astype(float)
    ggr_values = pd.to_numeric(frame[ggr_value_column], errors="raise").astype(float)

    positive_mask = y_true == 1
    negative_mask = y_true == 0
    predicted_positive_mask = predicted == 1

    row_count = int(len(frame))
    positive_label_count = int(y_true.sum())
    negative_label_count = int((y_true == 0).sum())
    predicted_positive_count = int(predicted.sum())
    created_at = datetime.utcnow().replace(microsecond=0).isoformat()

    return {
        "assignment_start_date": str(pt),
        "assignment_end_date": str(pt),
        "maturity_horizon_days": None,
        "status": "completed",
        "status_reason": None,
        "row_count": row_count,
        "evaluated_row_count": row_count,
        "positive_label_count": positive_label_count,
        "predicted_positive_count": predicted_positive_count,
        "true_positive_count": confusion["true_positive_count"],
        "false_positive_count": confusion["false_positive_count"],
        "false_negative_count": confusion["false_negative_count"],
        "true_negative_count": confusion["true_negative_count"],
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_score": float(f1_score(y_true, predicted, zero_division=0)),
        "response_rate_actual": float(y_true.mean()) if row_count else None,
        "response_rate_predicted": float(predicted.mean()) if row_count else None,
        "avg_score": float(scores.mean()) if row_count else None,
        "avg_score_positive_label": float(scores[positive_mask].mean()) if positive_label_count else None,
        "avg_score_negative_label": float(scores[negative_mask].mean()) if negative_label_count else None,
        "total_outcome_gross_bet_3d_value": float(bet_values.sum()) if row_count else None,
        "total_outcome_gross_ggr_3d_value": float(ggr_values.sum()) if row_count else None,
        "avg_outcome_gross_bet_3d_value": float(bet_values.mean()) if row_count else None,
        "avg_outcome_gross_ggr_3d_value": float(ggr_values.mean()) if row_count else None,
        "predicted_positive_total_gross_bet_3d_value": float(bet_values[predicted_positive_mask].sum()) if predicted_positive_count else 0.0,
        "predicted_positive_total_gross_ggr_3d_value": float(ggr_values[predicted_positive_mask].sum()) if predicted_positive_count else 0.0,
        "predicted_positive_avg_gross_bet_3d_value": float(bet_values[predicted_positive_mask].mean()) if predicted_positive_count else None,
        "predicted_positive_avg_gross_ggr_3d_value": float(ggr_values[predicted_positive_mask].mean()) if predicted_positive_count else None,
        "threshold": float(threshold),
        "model_name": _single_value_or_none(frame["model_name"]),
        "model_version": _single_value_or_none(frame["model_version"]),
        "created_at": created_at,
        "pt": str(pt),
    }


def build_not_evaluable_global_row(
    *,
    pt: str,
    status_reason: str,
) -> Dict[str, Any]:
    return {
        "assignment_start_date": str(pt),
        "assignment_end_date": str(pt),
        "maturity_horizon_days": None,
        "status": "not_evaluable",
        "status_reason": status_reason,
        "row_count": 0,
        "evaluated_row_count": 0,
        "positive_label_count": 0,
        "predicted_positive_count": 0,
        "true_positive_count": 0,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "true_negative_count": 0,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "response_rate_actual": None,
        "response_rate_predicted": None,
        "avg_score": None,
        "avg_score_positive_label": None,
        "avg_score_negative_label": None,
        "total_outcome_gross_bet_3d_value": None,
        "total_outcome_gross_ggr_3d_value": None,
        "avg_outcome_gross_bet_3d_value": None,
        "avg_outcome_gross_ggr_3d_value": None,
        "predicted_positive_total_gross_bet_3d_value": None,
        "predicted_positive_total_gross_ggr_3d_value": None,
        "predicted_positive_avg_gross_bet_3d_value": None,
        "predicted_positive_avg_gross_ggr_3d_value": None,
        "threshold": None,
        "model_name": None,
        "model_version": None,
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat(),
        "pt": str(pt),
    }


def build_performance_alerts(
    *,
    config: ResponseMonitoringConfig,
    global_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    now = datetime.utcnow().replace(microsecond=0).isoformat()
    thresholds = config.performance

    if int(global_row["row_count"]) < thresholds.min_rows:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "warning",
                "check_name": "insufficient_rows",
                "metric_name": "row_count",
                "observed_value": global_row["row_count"],
                "threshold_value": thresholds.min_rows,
                "reference_value": None,
                "message": "Delayed observational performance sample is smaller than the configured minimum.",
                "context": {},
            }
        )
    if int(global_row["positive_label_count"]) < thresholds.min_positive:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "warning",
                "check_name": "insufficient_positive_events",
                "metric_name": "positive_label_count",
                "observed_value": global_row["positive_label_count"],
                "threshold_value": thresholds.min_positive,
                "reference_value": None,
                "message": "Observed positive-response count is too small for stable delayed monitoring.",
                "context": {},
            }
        )

    response_rate = float(global_row["response_rate_actual"])
    if response_rate <= thresholds.response_rate_floor_alert:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "critical",
                "check_name": "realized_response_collapse",
                "metric_name": "response_rate_actual",
                "observed_value": response_rate,
                "threshold_value": thresholds.response_rate_floor_alert,
                "reference_value": None,
                "message": "Observed delayed response rate collapsed below the critical floor.",
                "context": {},
            }
        )
    elif response_rate <= thresholds.response_rate_floor_warn:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "warning",
                "check_name": "realized_response_collapse",
                "metric_name": "response_rate_actual",
                "observed_value": response_rate,
                "threshold_value": thresholds.response_rate_floor_warn,
                "reference_value": None,
                "message": "Observed delayed response rate fell below the warning floor.",
                "context": {},
            }
        )

    score_gap = abs(float(global_row["avg_score"]) - float(global_row["response_rate_actual"]))
    if score_gap >= thresholds.calibration_gap_alert:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "critical",
                "check_name": "calibration_gap",
                "metric_name": "avg_score_vs_actual_response_rate_gap",
                "observed_value": score_gap,
                "threshold_value": thresholds.calibration_gap_alert,
                "reference_value": None,
                "message": "Average predicted score and delayed observed response rate diverged materially.",
                "context": {},
            }
        )
    elif score_gap >= thresholds.calibration_gap_warn:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "warning",
                "check_name": "calibration_gap",
                "metric_name": "avg_score_vs_actual_response_rate_gap",
                "observed_value": score_gap,
                "threshold_value": thresholds.calibration_gap_warn,
                "reference_value": None,
                "message": "Average predicted score and delayed observed response rate are diverging.",
                "context": {},
            }
        )
    return alerts


def build_performance_markdown_report(
    *,
    pt: str,
    maturity_summary: Dict[str, Any],
    global_row: Dict[str, Any],
    alerts: List[Dict[str, Any]],
) -> List[str]:
    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [
        "# Phase-1 Response Delayed Observational Monitoring",
        "",
        f"- scoring partition: `{pt}`",
        f"- score date: `{maturity_summary['score_date']}`",
        f"- observed outcome end date: `{maturity_summary['observed_outcome_end_date']}`",
        f"- matured on date: `{maturity_summary['matured_on_date']}`",
        f"- labels mature: `{maturity_summary.get('labels_mature')}`",
        f"- status: `{global_row['status']}`",
        f"- rows / evaluated rows: `{global_row['row_count']}` / `{global_row['evaluated_row_count']}`",
        f"- positive label count / rate: `{global_row['positive_label_count']}` / `{_fmt(global_row['response_rate_actual'])}`",
        f"- predicted positive count / rate: `{global_row['predicted_positive_count']}` / `{_fmt(global_row['response_rate_predicted'])}`",
        f"- threshold: `{_fmt(global_row['threshold'])}`",
        f"- precision / recall / f1: `{_fmt(global_row['precision'])}` / `{_fmt(global_row['recall'])}` / `{_fmt(global_row['f1_score'])}`",
        f"- average score: `{_fmt(global_row['avg_score'])}`",
        f"- total gross bet / gross ggr: `{_fmt(global_row['total_outcome_gross_bet_3d_value'])}` / `{_fmt(global_row['total_outcome_gross_ggr_3d_value'])}`",
        "",
        "## Alerts",
        "",
    ]
    if not alerts:
        lines.append("- none")
    else:
        for alert in alerts:
            lines.append(
                f"- `{alert['severity']}` `{alert['check_name']}` `{alert['metric_name']}` observed=`{alert['observed_value']}` threshold=`{alert['threshold_value']}`: {alert['message']}"
            )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- This report measures delayed observed response after scoring-date anchoring. It is operational observational monitoring, not causal uplift validation and not proof of incremental effect.",
        ]
    )
    return lines
