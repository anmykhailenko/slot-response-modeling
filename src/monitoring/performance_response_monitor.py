from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

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


def validate_observational_frame(frame: pd.DataFrame, *, response_window_days: int) -> None:
    required = {
        "player_id",
        "pt",
        "predicted_response_score",
        "response_priority_bucket",
        "selected_threshold",
        f"observed_response_label_positive_{response_window_days}d",
        f"observed_gross_bet_value_{response_window_days}d",
    }
    missing = [column for column in sorted(required) if column not in frame.columns]
    if missing:
        raise ValueError(f"Delayed response monitoring input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Delayed response monitoring input is empty.")
    duplicate_count = int(frame.duplicated(subset=["player_id", "pt"]).sum())
    if duplicate_count:
        raise ValueError(f"Delayed response monitoring input contains duplicate player_id,pt rows: {duplicate_count}")


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    threshold = float(np.quantile(scores, 0.8)) if len(scores) else 0.5
    predicted = (scores >= threshold).astype(int)
    summary = {
        "row_count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "observed_positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "avg_predicted_score": float(scores.mean()) if len(scores) else 0.0,
        "predicted_positive_rate": float(predicted.mean()) if len(predicted) else 0.0,
        "calibration_gap": float(abs(scores.mean() - y_true.mean())) if len(scores) else 0.0,
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
    }
    if len(np.unique(y_true)) > 1:
        summary["pr_auc"] = float(average_precision_score(y_true, scores))
        summary["roc_auc"] = float(roc_auc_score(y_true, scores))
        summary["log_loss"] = float(log_loss(y_true, scores, labels=[0, 1]))
        summary["brier_score"] = float(brier_score_loss(y_true, scores))
    else:
        summary["pr_auc"] = 0.0
        summary["roc_auc"] = 0.0
        summary["log_loss"] = 0.0
        summary["brier_score"] = float(brier_score_loss(y_true, scores)) if len(scores) else 0.0
    return summary


def build_global_performance_row(frame: pd.DataFrame, *, pt: str, response_window_days: int) -> Dict[str, Any]:
    label_column = f"observed_response_label_positive_{response_window_days}d"
    value_column = f"observed_gross_bet_value_{response_window_days}d"
    y_true = pd.to_numeric(frame[label_column], errors="raise").astype(int).to_numpy()
    scores = pd.to_numeric(frame["predicted_response_score"], errors="raise").astype(float).to_numpy()
    summary = _metrics(y_true, scores)
    summary.update(
        {
            "monitor_run_ts": datetime.utcnow().replace(microsecond=0).isoformat(),
            "metric_scope": "global",
            "segment_column": None,
            "segment_value": None,
            "bucket_name": None,
            "avg_observed_gross_bet_value": float(pd.to_numeric(frame[value_column], errors="raise").mean()),
            "status": "evaluated",
            "status_reason": None,
            "pt": str(pt),
        }
    )
    return summary


def build_bucket_performance_rows(frame: pd.DataFrame, *, pt: str, response_window_days: int, min_rows: int) -> List[Dict[str, Any]]:
    label_column = f"observed_response_label_positive_{response_window_days}d"
    value_column = f"observed_gross_bet_value_{response_window_days}d"
    rows: List[Dict[str, Any]] = []
    for bucket_name, bucket_frame in frame.groupby("response_priority_bucket", dropna=False, observed=True):
        if len(bucket_frame) < min_rows:
            rows.append(
                {
                    "monitor_run_ts": datetime.utcnow().replace(microsecond=0).isoformat(),
                    "metric_scope": "bucket",
                    "segment_column": None,
                    "segment_value": None,
                    "bucket_name": "__missing__" if pd.isna(bucket_name) else str(bucket_name),
                    "row_count": int(len(bucket_frame)),
                    "positive_count": int(pd.to_numeric(bucket_frame[label_column], errors="raise").sum()),
                    "observed_positive_rate": float(pd.to_numeric(bucket_frame[label_column], errors="raise").mean()) if len(bucket_frame) else 0.0,
                    "avg_predicted_score": float(pd.to_numeric(bucket_frame["predicted_response_score"], errors="raise").mean()) if len(bucket_frame) else 0.0,
                    "predicted_positive_rate": None,
                    "calibration_gap": None,
                    "pr_auc": None,
                    "roc_auc": None,
                    "log_loss": None,
                    "brier_score": None,
                    "score_min": float(pd.to_numeric(bucket_frame["predicted_response_score"], errors="raise").min()) if len(bucket_frame) else None,
                    "score_max": float(pd.to_numeric(bucket_frame["predicted_response_score"], errors="raise").max()) if len(bucket_frame) else None,
                    "avg_observed_gross_bet_value": float(pd.to_numeric(bucket_frame[value_column], errors="raise").mean()) if len(bucket_frame) else 0.0,
                    "status": "not_evaluable",
                    "status_reason": "insufficient_rows",
                    "pt": str(pt),
                }
            )
            continue
        y_true = pd.to_numeric(bucket_frame[label_column], errors="raise").astype(int).to_numpy()
        scores = pd.to_numeric(bucket_frame["predicted_response_score"], errors="raise").astype(float).to_numpy()
        summary = _metrics(y_true, scores)
        summary.update(
            {
                "monitor_run_ts": datetime.utcnow().replace(microsecond=0).isoformat(),
                "metric_scope": "bucket",
                "segment_column": None,
                "segment_value": None,
                "bucket_name": "__missing__" if pd.isna(bucket_name) else str(bucket_name),
                "avg_observed_gross_bet_value": float(pd.to_numeric(bucket_frame[value_column], errors="raise").mean()),
                "status": "evaluated",
                "status_reason": None,
                "pt": str(pt),
            }
        )
        rows.append(summary)
    return rows


def build_segment_performance_rows(
    frame: pd.DataFrame,
    *,
    pt: str,
    response_window_days: int,
    segment_columns: Sequence[str],
    min_rows: int,
) -> List[Dict[str, Any]]:
    label_column = f"observed_response_label_positive_{response_window_days}d"
    value_column = f"observed_gross_bet_value_{response_window_days}d"
    rows: List[Dict[str, Any]] = []
    for segment_column in segment_columns:
        if segment_column not in frame.columns:
            continue
        working = frame.copy()
        working[segment_column] = working[segment_column].fillna("__missing__").astype(str)
        for segment_value, segment_frame in working.groupby(segment_column, dropna=False, observed=True):
            if len(segment_frame) < min_rows:
                rows.append(
                    {
                        "monitor_run_ts": datetime.utcnow().replace(microsecond=0).isoformat(),
                        "metric_scope": "segment",
                        "segment_column": segment_column,
                        "segment_value": str(segment_value),
                        "bucket_name": None,
                        "row_count": int(len(segment_frame)),
                        "positive_count": int(pd.to_numeric(segment_frame[label_column], errors="raise").sum()),
                        "observed_positive_rate": float(pd.to_numeric(segment_frame[label_column], errors="raise").mean()) if len(segment_frame) else 0.0,
                        "avg_predicted_score": float(pd.to_numeric(segment_frame["predicted_response_score"], errors="raise").mean()) if len(segment_frame) else 0.0,
                        "predicted_positive_rate": None,
                        "calibration_gap": None,
                        "pr_auc": None,
                        "roc_auc": None,
                        "log_loss": None,
                        "brier_score": None,
                        "score_min": float(pd.to_numeric(segment_frame["predicted_response_score"], errors="raise").min()) if len(segment_frame) else None,
                        "score_max": float(pd.to_numeric(segment_frame["predicted_response_score"], errors="raise").max()) if len(segment_frame) else None,
                        "avg_observed_gross_bet_value": float(pd.to_numeric(segment_frame[value_column], errors="raise").mean()) if len(segment_frame) else 0.0,
                        "status": "not_evaluable",
                        "status_reason": "insufficient_rows",
                        "pt": str(pt),
                    }
                )
                continue
            y_true = pd.to_numeric(segment_frame[label_column], errors="raise").astype(int).to_numpy()
            scores = pd.to_numeric(segment_frame["predicted_response_score"], errors="raise").astype(float).to_numpy()
            summary = _metrics(y_true, scores)
            summary.update(
                {
                    "monitor_run_ts": datetime.utcnow().replace(microsecond=0).isoformat(),
                    "metric_scope": "segment",
                    "segment_column": segment_column,
                    "segment_value": str(segment_value),
                    "bucket_name": None,
                    "avg_observed_gross_bet_value": float(pd.to_numeric(segment_frame[value_column], errors="raise").mean()),
                    "status": "evaluated",
                    "status_reason": None,
                    "pt": str(pt),
                }
            )
            rows.append(summary)
    return rows


def build_calibration_rows(frame: pd.DataFrame, *, pt: str, response_window_days: int, bins: int = 10) -> List[Dict[str, Any]]:
    label_column = f"observed_response_label_positive_{response_window_days}d"
    working = frame.copy()
    working["predicted_response_score"] = pd.to_numeric(working["predicted_response_score"], errors="raise").astype(float)
    working[label_column] = pd.to_numeric(working[label_column], errors="raise").astype(int)
    if working.empty:
        return []
    quantile_count = min(bins, len(working))
    working["calibration_bin"] = pd.qcut(working["predicted_response_score"].rank(method="first"), q=quantile_count, duplicates="drop")
    rows: List[Dict[str, Any]] = []
    for score_bin, bin_frame in working.groupby("calibration_bin", dropna=False, observed=False):
        rows.append(
            {
                "pt": str(pt),
                "calibration_bin": str(score_bin),
                "row_count": int(len(bin_frame)),
                "average_predicted_score": float(bin_frame["predicted_response_score"].mean()),
                "actual_positive_rate": float(bin_frame[label_column].mean()),
                "calibration_gap": float(abs(bin_frame["predicted_response_score"].mean() - bin_frame[label_column].mean())),
            }
        )
    return rows


def build_performance_alerts(
    *,
    config: ResponseMonitoringConfig,
    global_row: Dict[str, Any],
    calibration_rows: List[Dict[str, Any]],
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
    if int(global_row["positive_count"]) < thresholds.min_positive:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "warning",
                "check_name": "insufficient_positive_events",
                "metric_name": "positive_count",
                "observed_value": global_row["positive_count"],
                "threshold_value": thresholds.min_positive,
                "reference_value": None,
                "message": "Observed positive-response count is too small for stable delayed monitoring.",
                "context": {},
            }
        )
    response_rate = float(global_row["observed_positive_rate"])
    if response_rate <= thresholds.response_rate_floor_alert:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "critical",
                "check_name": "realized_response_collapse",
                "metric_name": "observed_positive_rate",
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
                "metric_name": "observed_positive_rate",
                "observed_value": response_rate,
                "threshold_value": thresholds.response_rate_floor_warn,
                "reference_value": None,
                "message": "Observed delayed response rate fell below the warning floor.",
                "context": {},
            }
        )
    calibration_gap = float(global_row["calibration_gap"])
    if calibration_gap >= thresholds.calibration_gap_alert:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "critical",
                "check_name": "calibration_gap",
                "metric_name": "global_calibration_gap",
                "observed_value": calibration_gap,
                "threshold_value": thresholds.calibration_gap_alert,
                "reference_value": None,
                "message": "Average predicted score and delayed observed response rate diverged materially.",
                "context": {},
            }
        )
    elif calibration_gap >= thresholds.calibration_gap_warn:
        alerts.append(
            {
                "monitor_run_ts": now,
                "severity": "warning",
                "check_name": "calibration_gap",
                "metric_name": "global_calibration_gap",
                "observed_value": calibration_gap,
                "threshold_value": thresholds.calibration_gap_warn,
                "reference_value": None,
                "message": "Average predicted score and delayed observed response rate are diverging.",
                "context": {},
            }
        )
    if calibration_rows:
        worst_gap = max(float(row["calibration_gap"]) for row in calibration_rows)
        if worst_gap >= thresholds.calibration_gap_alert:
            alerts.append(
                {
                    "monitor_run_ts": now,
                    "severity": "warning",
                    "check_name": "bin_level_calibration_gap",
                    "metric_name": "max_bin_calibration_gap",
                    "observed_value": worst_gap,
                    "threshold_value": thresholds.calibration_gap_alert,
                    "reference_value": None,
                    "message": "At least one calibration bin shows materially weak alignment between score and observed response.",
                    "context": {},
                }
            )
    return alerts


def build_performance_markdown_report(
    *,
    pt: str,
    maturity_summary: Dict[str, Any],
    global_row: Dict[str, Any],
    bucket_rows: List[Dict[str, Any]],
    segment_rows: List[Dict[str, Any]],
    calibration_rows: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
) -> List[str]:
    lines = [
        "# Phase-1 Response Delayed Observational Monitoring",
        "",
        f"- scoring partition: `{pt}`",
        f"- score date: `{maturity_summary['score_date']}`",
        f"- observed outcome end date: `{maturity_summary['observed_outcome_end_date']}`",
        f"- matured on date: `{maturity_summary['matured_on_date']}`",
        f"- global rows: `{global_row['row_count']}`",
        f"- observed positive count / rate: `{global_row['positive_count']}` / `{global_row['observed_positive_rate']:.6f}`",
        f"- average predicted score: `{global_row['avg_predicted_score']:.6f}`",
        f"- calibration gap: `{global_row['calibration_gap']:.6f}`",
        f"- PR AUC / ROC AUC: `{global_row['pr_auc']:.6f}` / `{global_row['roc_auc']:.6f}`",
        "",
        "## Bucket-Level Observational Performance",
        "",
        "| bucket | rows | observed_positive_rate | avg_predicted_score | calibration_gap | status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in bucket_rows:
        gap = "n/a" if row["calibration_gap"] is None else f"{row['calibration_gap']:.6f}"
        lines.append(
            f"| {row['bucket_name']} | {row['row_count']} | {row['observed_positive_rate']:.6f} | {row['avg_predicted_score']:.6f} | {gap} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Segment-Level Observational Performance",
            "",
            "| segment_column | segment_value | rows | observed_positive_rate | avg_predicted_score | status |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in segment_rows:
        lines.append(
            f"| {row['segment_column']} | {row['segment_value']} | {row['row_count']} | {row['observed_positive_rate']:.6f} | {row['avg_predicted_score']:.6f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Calibration Summary",
            "",
            "| bin | rows | average_predicted_score | actual_positive_rate | calibration_gap |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in calibration_rows:
        lines.append(
            f"| {row['calibration_bin']} | {row['row_count']} | {row['average_predicted_score']:.6f} | {row['actual_positive_rate']:.6f} | {row['calibration_gap']:.6f} |"
        )
    lines.extend(["", "## Alerts", ""])
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
