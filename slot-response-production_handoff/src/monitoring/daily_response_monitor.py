from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .config import ResponseMonitoringConfig
from .odps_io import serialize_json


REQUIRED_SCORED_COLUMNS = [
    "player_id",
    "score_date",
    "scoring_pt",
    "scoring_ts",
    "snapshot_date",
    "predicted_response_score",
    "score_rank",
    "score_percentile",
    "response_priority_bucket",
    "action_recommendation",
    "vip_level",
    "model_name",
    "model_version",
    "model_reference_path",
    "selected_threshold",
    "pt",
]


def required_scored_columns() -> List[str]:
    return list(REQUIRED_SCORED_COLUMNS)


def validate_scored_frame(scored: pd.DataFrame, *, expected_pt: str) -> Dict[str, Any]:
    missing = [column for column in REQUIRED_SCORED_COLUMNS if column not in scored.columns]
    if missing:
        raise ValueError(f"Response monitoring input is missing required columns: {missing}")
    if scored.empty:
        raise ValueError(f"Response monitoring input is empty for pt={expected_pt}.")
    duplicate_player_count = int(scored.duplicated(subset=["player_id"]).sum())
    duplicate_partition_count = int(scored.duplicated(subset=["player_id", "pt"]).sum())
    invalid_score_count = int(
        scored["predicted_response_score"].isna().sum()
        + ((pd.to_numeric(scored["predicted_response_score"], errors="coerce") < 0.0) | (pd.to_numeric(scored["predicted_response_score"], errors="coerce") > 1.0)).sum()
    )
    pts = sorted(scored["pt"].dropna().astype(str).unique().tolist())
    scoring_pts = sorted(scored["scoring_pt"].dropna().astype(str).unique().tolist())
    if pts != [expected_pt]:
        raise ValueError(f"Monitored score partition contains unexpected `pt` values: {pts}; expected `{expected_pt}`.")
    if scoring_pts != [expected_pt]:
        raise ValueError(f"Monitored score partition contains unexpected `scoring_pt` values: {scoring_pts}; expected `{expected_pt}`.")
    return {
        "duplicate_player_count": duplicate_player_count,
        "duplicate_partition_count": duplicate_partition_count,
        "invalid_score_count": invalid_score_count,
    }


def compute_score_distribution(scored: pd.DataFrame) -> Dict[str, float]:
    values = pd.to_numeric(scored["predicted_response_score"], errors="coerce")
    quantiles = {name: float(values.quantile(q)) for name, q in [("p01", 0.01), ("p05", 0.05), ("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]}
    return {
        "score_min": float(values.min()),
        "score_max": float(values.max()),
        "score_mean": float(values.mean()),
        "score_median": float(values.median()),
        "score_std": float(values.std()),
        **quantiles,
    }


def build_score_distribution_for_frame(scored: pd.DataFrame) -> Dict[str, float]:
    return compute_score_distribution(scored)


def build_prediction_volume_summary(scored: pd.DataFrame, *, selected_threshold: Optional[float]) -> Dict[str, Any]:
    if selected_threshold is None or scored.empty:
        return {
            "positive_prediction_count": 0,
            "negative_prediction_count": 0,
            "predicted_positive_rate": 0.0,
            "predicted_negative_rate": 0.0,
        }
    predicted_positive = pd.to_numeric(scored["predicted_response_score"], errors="coerce") >= float(selected_threshold)
    positive_prediction_count = int(predicted_positive.sum())
    predicted_positive_rate = float(predicted_positive.mean()) if len(predicted_positive) else 0.0
    return {
        "positive_prediction_count": positive_prediction_count,
        "negative_prediction_count": int(len(scored) - positive_prediction_count),
        "predicted_positive_rate": predicted_positive_rate,
        "predicted_negative_rate": float(1.0 - predicted_positive_rate),
    }


def build_bucket_distribution(scored: pd.DataFrame) -> List[Dict[str, Any]]:
    bucket_rows = (
        scored.groupby("response_priority_bucket", as_index=False, dropna=False, observed=True)
        .agg(
            row_count=("player_id", "count"),
            distinct_player_count=("player_id", "nunique"),
            score_min=("predicted_response_score", "min"),
            score_max=("predicted_response_score", "max"),
            score_mean=("predicted_response_score", "mean"),
        )
        .sort_values(["row_count", "response_priority_bucket"], ascending=[False, True])
    )
    total_rows = int(bucket_rows["row_count"].sum()) if not bucket_rows.empty else 0
    rows: List[Dict[str, Any]] = []
    for row in bucket_rows.to_dict("records"):
        rows.append(
            {
                "bucket_name": row["response_priority_bucket"],
                "row_count": int(row["row_count"]),
                "distinct_player_count": int(row["distinct_player_count"]),
                "row_share": float(row["row_count"] / total_rows) if total_rows else 0.0,
                "score_min": float(row["score_min"]),
                "score_max": float(row["score_max"]),
                "score_mean": float(row["score_mean"]),
            }
        )
    return rows


def build_segment_distribution(scored: pd.DataFrame, *, segment_column: str) -> List[Dict[str, Any]]:
    if segment_column not in scored.columns:
        return []
    working = scored.copy()
    working[segment_column] = working[segment_column].fillna("__missing__").astype(str)
    grouped = (
        working.groupby(segment_column, as_index=False, dropna=False, observed=True)
        .agg(
            row_count=("player_id", "count"),
            distinct_player_count=("player_id", "nunique"),
            avg_score=("predicted_response_score", "mean"),
            median_score=("predicted_response_score", "median"),
        )
        .sort_values(["row_count", segment_column], ascending=[False, True])
    )
    total_rows = int(grouped["row_count"].sum()) if not grouped.empty else 0
    rows: List[Dict[str, Any]] = []
    for row in grouped.to_dict("records"):
        rows.append(
            {
                "segment_column": segment_column,
                "segment_value": row[segment_column],
                "row_count": int(row["row_count"]),
                "distinct_player_count": int(row["distinct_player_count"]),
                "row_share": float(row["row_count"] / total_rows) if total_rows else 0.0,
                "avg_score": float(row["avg_score"]),
                "median_score": float(row["median_score"]),
            }
        )
    return rows


def build_score_by_segment(scored: pd.DataFrame, *, segment_column: str) -> List[Dict[str, Any]]:
    if segment_column not in scored.columns:
        return []
    working = scored.copy()
    working[segment_column] = working[segment_column].fillna("__missing__").astype(str)
    grouped = (
        working.groupby(["response_priority_bucket", segment_column], as_index=False, dropna=False, observed=True)
        .agg(
            row_count=("player_id", "count"),
            avg_score=("predicted_response_score", "mean"),
        )
        .sort_values(["response_priority_bucket", "row_count"], ascending=[True, False])
    )
    return [
        {
            "bucket_name": row["response_priority_bucket"],
            "segment_column": segment_column,
            "segment_value": row[segment_column],
            "row_count": int(row["row_count"]),
            "avg_score": float(row["avg_score"]),
        }
        for row in grouped.to_dict("records")
    ]


def build_daily_segment_summary_rows(
    *,
    config: ResponseMonitoringConfig,
    scored: pd.DataFrame,
    pt: str,
    reference_pt: Optional[str],
    segment_columns: Sequence[str],
    model_name: Optional[str],
    model_version: Optional[str],
    model_reference_path: Optional[str],
    selected_threshold: Optional[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for segment_column in segment_columns:
        if segment_column not in scored.columns:
            continue
        working = scored.copy()
        working[segment_column] = working[segment_column].fillna("__missing__").astype(str)
        for segment_value, segment_frame in working.groupby(segment_column, dropna=False, observed=False):
            distribution = build_score_distribution_for_frame(segment_frame)
            bucket_distribution = build_bucket_distribution(segment_frame)
            snapshot_dates = pd.to_datetime(segment_frame["snapshot_date"], errors="coerce")
            score_dates = pd.to_datetime(segment_frame["score_date"], errors="coerce")
            snapshot_date_min = snapshot_dates.min().date().isoformat() if snapshot_dates.notna().any() else None
            snapshot_date_max = snapshot_dates.max().date().isoformat() if snapshot_dates.notna().any() else None
            snapshot_date_distinct_count = int(snapshot_dates.dt.date.nunique()) if snapshot_dates.notna().any() else 0
            snapshot_date_lag_days = None
            if snapshot_dates.notna().any() and score_dates.notna().any():
                snapshot_date_lag_days = int((score_dates.max().date() - snapshot_dates.max().date()).days)
            rows.append(
                {
                    "monitor_run_ts": datetime.utcnow().replace(microsecond=0).isoformat(),
                    "run_label": None,
                    "metric_scope": "segment",
                    "segment_column": segment_column,
                    "segment_value": str(segment_value),
                    "monitor_status": "ok",
                    "scoring_table": config.source.scored_table,
                    "reference_pt": reference_pt,
                    "partition_present_flag": 1,
                    "write_success_flag": 1,
                    "row_count": int(len(segment_frame)),
                    "distinct_player_count": int(segment_frame["player_id"].astype("string").nunique()),
                    "duplicate_player_count": int(segment_frame.duplicated(subset=["player_id"]).sum()),
                    "duplicate_partition_count": int(segment_frame.duplicated(subset=["player_id", "pt"]).sum()),
                    "invalid_score_count": int(
                        segment_frame["predicted_response_score"].isna().sum()
                        + ((pd.to_numeric(segment_frame["predicted_response_score"], errors="coerce") < 0.0) | (pd.to_numeric(segment_frame["predicted_response_score"], errors="coerce") > 1.0)).sum()
                    ),
                    "metadata_missing_rows": int(
                        segment_frame[["model_name", "model_version", "model_reference_path", "selected_threshold"]].isna().any(axis=1).sum()
                    ),
                    "model_name": model_name,
                    "model_version": model_version,
                    "model_reference_path": model_reference_path,
                    "selected_threshold": selected_threshold,
                    "snapshot_date_min": snapshot_date_min,
                    "snapshot_date_max": snapshot_date_max,
                    "snapshot_date_distinct_count": snapshot_date_distinct_count,
                    "snapshot_date_lag_days": snapshot_date_lag_days,
                    **distribution,
                    "bucket_distribution_json": serialize_json(bucket_distribution),
                    "vip_distribution_json": None,
                    "score_by_segment_json": None,
                    "eligible_population_json": None,
                    "alert_count": 0,
                    "alerts_json": serialize_json([]),
                    "pt": str(pt),
                }
            )
    return rows


def build_population_summary(
    *,
    current_eligible: Optional[pd.DataFrame],
    reference_eligible: Optional[pd.DataFrame],
    segment_column: str,
) -> List[Dict[str, Any]]:
    if current_eligible is None:
        return []
    current = current_eligible.rename(columns={"segment_value": "segment_value_current"})
    reference = None if reference_eligible is None else reference_eligible.rename(columns={"segment_value": "segment_value_reference"})
    if reference is None:
        return [
            {
                "segment_column": segment_column,
                "segment_value": row["segment_value_current"],
                "eligible_row_count": int(row["eligible_row_count"]),
                "eligible_distinct_player_count": int(row["eligible_distinct_player_count"]),
                "reference_eligible_row_count": None,
                "eligible_row_delta": None,
                "eligible_row_delta_ratio": None,
                "snapshot_date_min": row.get("snapshot_date_min"),
                "snapshot_date_max": row.get("snapshot_date_max"),
            }
            for row in current.to_dict("records")
        ]
    merged = current.merge(
        reference,
        how="outer",
        left_on="segment_value_current",
        right_on="segment_value_reference",
        suffixes=("_current", "_reference"),
    ).fillna({"eligible_row_count_current": 0, "eligible_row_count_reference": 0, "eligible_distinct_player_count_current": 0, "eligible_distinct_player_count_reference": 0})
    rows: List[Dict[str, Any]] = []
    for row in merged.to_dict("records"):
        current_count = int(row.get("eligible_row_count_current", 0))
        reference_count = int(row.get("eligible_row_count_reference", 0))
        rows.append(
            {
                "segment_column": segment_column,
                "segment_value": row.get("segment_value_current") or row.get("segment_value_reference"),
                "eligible_row_count": current_count,
                "eligible_distinct_player_count": int(row.get("eligible_distinct_player_count_current", 0)),
                "reference_eligible_row_count": reference_count,
                "eligible_row_delta": current_count - reference_count,
                "eligible_row_delta_ratio": float((current_count - reference_count) / reference_count) if reference_count else None,
                "snapshot_date_min": row.get("snapshot_date_min_current"),
                "snapshot_date_max": row.get("snapshot_date_max_current"),
            }
        )
    return rows


def _build_alert(
    *,
    severity: str,
    check_name: str,
    metric_name: str,
    observed_value: Any,
    threshold_value: Any,
    message: str,
    reference_value: Any = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "monitor_run_ts": datetime.utcnow().replace(microsecond=0).isoformat(),
        "severity": severity,
        "check_name": check_name,
        "metric_name": metric_name,
        "observed_value": observed_value,
        "threshold_value": threshold_value,
        "reference_value": reference_value,
        "message": message,
        "context": context or {},
    }


def build_daily_alerts(
    *,
    config: ResponseMonitoringConfig,
    row_count: int,
    reference_row_count: Optional[int],
    validation_summary: Dict[str, Any],
    score_distribution: Dict[str, Any],
    bucket_distribution: List[Dict[str, Any]],
    vip_distribution: List[Dict[str, Any]],
    metadata_missing_rows: int,
    snapshot_date_lag_days: Optional[int],
    snapshot_date_distinct_count: int,
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    thresholds = config.daily
    if row_count < thresholds.min_row_count:
        alerts.append(
            _build_alert(
                severity="critical",
                check_name="empty_or_small_partition",
                metric_name="row_count",
                observed_value=row_count,
                threshold_value=thresholds.min_row_count,
                message="Scored response partition is empty or below minimum expected size.",
            )
        )
    if validation_summary["duplicate_player_count"] >= thresholds.duplicate_player_alert_count:
        alerts.append(
            _build_alert(
                severity="critical",
                check_name="duplicate_player_ids",
                metric_name="duplicate_player_count",
                observed_value=validation_summary["duplicate_player_count"],
                threshold_value=thresholds.duplicate_player_alert_count,
                message="Duplicate player_id rows detected in the scored partition.",
            )
        )
    if validation_summary["invalid_score_count"] >= thresholds.invalid_score_alert_count:
        alerts.append(
            _build_alert(
                severity="critical",
                check_name="invalid_score_range",
                metric_name="invalid_score_count",
                observed_value=validation_summary["invalid_score_count"],
                threshold_value=thresholds.invalid_score_alert_count,
                message="Null or out-of-range predicted response scores were detected.",
            )
        )
    if metadata_missing_rows > thresholds.max_missing_metadata_rows:
        alerts.append(
            _build_alert(
                severity="critical",
                check_name="missing_model_metadata",
                metric_name="metadata_missing_rows",
                observed_value=metadata_missing_rows,
                threshold_value=thresholds.max_missing_metadata_rows,
                message="Model metadata is missing for one or more scored rows.",
            )
        )
    if snapshot_date_lag_days is not None and snapshot_date_lag_days > thresholds.max_snapshot_date_lag_days:
        alerts.append(
            _build_alert(
                severity="warning",
                check_name="snapshot_date_staleness",
                metric_name="snapshot_date_lag_days",
                observed_value=snapshot_date_lag_days,
                threshold_value=thresholds.max_snapshot_date_lag_days,
                message="Feature snapshot date appears stale relative to the scoring partition.",
            )
        )
    if snapshot_date_distinct_count > thresholds.max_snapshot_date_distinct_count:
        alerts.append(
            _build_alert(
                severity="warning",
                check_name="snapshot_date_dispersion",
                metric_name="snapshot_date_distinct_count",
                observed_value=snapshot_date_distinct_count,
                threshold_value=thresholds.max_snapshot_date_distinct_count,
                message="Multiple snapshot dates were observed in one scored partition.",
            )
        )
    if reference_row_count:
        ratio = (row_count - reference_row_count) / reference_row_count
        if ratio <= -thresholds.row_count_drop_alert_ratio:
            alerts.append(
                _build_alert(
                    severity="critical",
                    check_name="row_count_drop",
                    metric_name="row_count_delta_ratio",
                    observed_value=ratio,
                    threshold_value=-thresholds.row_count_drop_alert_ratio,
                    reference_value=reference_row_count,
                    message="Scored row count dropped materially versus the reference partition.",
                )
            )
        elif ratio <= -thresholds.row_count_drop_warn_ratio:
            alerts.append(
                _build_alert(
                    severity="warning",
                    check_name="row_count_drop",
                    metric_name="row_count_delta_ratio",
                    observed_value=ratio,
                    threshold_value=-thresholds.row_count_drop_warn_ratio,
                    reference_value=reference_row_count,
                    message="Scored row count declined versus the reference partition.",
                )
            )
        if ratio >= thresholds.row_count_growth_alert_ratio:
            alerts.append(
                _build_alert(
                    severity="warning",
                    check_name="row_count_growth",
                    metric_name="row_count_delta_ratio",
                    observed_value=ratio,
                    threshold_value=thresholds.row_count_growth_alert_ratio,
                    reference_value=reference_row_count,
                    message="Scored row count grew sharply versus the reference partition.",
                )
            )
        elif ratio >= thresholds.row_count_growth_warn_ratio:
            alerts.append(
                _build_alert(
                    severity="warning",
                    check_name="row_count_growth",
                    metric_name="row_count_delta_ratio",
                    observed_value=ratio,
                    threshold_value=thresholds.row_count_growth_warn_ratio,
                    reference_value=reference_row_count,
                    message="Scored row count increased meaningfully versus the reference partition.",
                )
            )
    top_bucket_share = 0.0
    for row in bucket_distribution:
        if str(row["bucket_name"]) in {"very_high", "high"}:
            top_bucket_share += float(row["row_share"])
    if top_bucket_share < thresholds.min_top_bucket_share or top_bucket_share > thresholds.max_top_bucket_share:
        alerts.append(
            _build_alert(
                severity="warning",
                check_name="top_bucket_share",
                metric_name="top_bucket_share",
                observed_value=top_bucket_share,
                threshold_value=f"[{thresholds.min_top_bucket_share}, {thresholds.max_top_bucket_share}]",
                message="The share of high-priority buckets is outside the configured operating range.",
            )
        )
    vip_missing_rate = next((float(row["row_share"]) for row in vip_distribution if str(row["segment_value"]) == "__missing__"), 0.0)
    if vip_missing_rate > thresholds.max_vip_missing_rate:
        alerts.append(
            _build_alert(
                severity="warning",
                check_name="vip_level_missingness",
                metric_name="vip_missing_rate",
                observed_value=vip_missing_rate,
                threshold_value=thresholds.max_vip_missing_rate,
                message="VIP level missingness exceeded the configured limit.",
            )
        )
    if float(score_distribution["score_min"]) < 0.0 or float(score_distribution["score_max"]) > 1.0:
        alerts.append(
            _build_alert(
                severity="critical",
                check_name="score_range",
                metric_name="score_min_max",
                observed_value=f"{score_distribution['score_min']}..{score_distribution['score_max']}",
                threshold_value="[0.0, 1.0]",
                message="Predicted response scores fell outside the valid probability range.",
            )
        )
    return alerts


def build_daily_summary_row(
    *,
    config: ResponseMonitoringConfig,
    pt: str,
    reference_pt: Optional[str],
    row_count: int,
    distinct_player_count: int,
    validation_summary: Dict[str, Any],
    score_distribution: Dict[str, Any],
    bucket_distribution: List[Dict[str, Any]],
    vip_distribution: List[Dict[str, Any]],
    score_by_segment_rows: List[Dict[str, Any]],
    eligible_population_rows: List[Dict[str, Any]],
    metadata_missing_rows: int,
    model_name: Optional[str],
    model_version: Optional[str],
    model_reference_path: Optional[str],
    selected_threshold: Optional[float],
    prediction_volume_summary: Dict[str, Any],
    snapshot_date_min: Any,
    snapshot_date_max: Any,
    snapshot_date_distinct_count: int,
    snapshot_date_lag_days: Optional[int],
    partition_present_flag: int,
    alerts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    overall_status = "ok"
    if any(alert["severity"] == "critical" for alert in alerts):
        overall_status = "critical"
    elif any(alert["severity"] == "warning" for alert in alerts):
        overall_status = "warning"
    return {
        "monitor_run_ts": datetime.utcnow().replace(microsecond=0).isoformat(),
        "run_label": None,
        "metric_scope": "global",
        "segment_column": None,
        "segment_value": None,
        "monitor_status": overall_status,
        "scoring_table": config.source.scored_table,
        "reference_pt": reference_pt,
        "partition_present_flag": int(partition_present_flag),
        "write_success_flag": 1,
        "row_count": int(row_count),
        "distinct_player_count": int(distinct_player_count),
        "duplicate_player_count": int(validation_summary["duplicate_player_count"]),
        "duplicate_partition_count": int(validation_summary["duplicate_partition_count"]),
        "invalid_score_count": int(validation_summary["invalid_score_count"]),
        "metadata_missing_rows": int(metadata_missing_rows),
        "model_name": model_name,
        "model_version": model_version,
        "model_reference_path": model_reference_path,
        "selected_threshold": selected_threshold,
        **prediction_volume_summary,
        "snapshot_date_min": snapshot_date_min,
        "snapshot_date_max": snapshot_date_max,
        "snapshot_date_distinct_count": int(snapshot_date_distinct_count),
        "snapshot_date_lag_days": snapshot_date_lag_days,
        **score_distribution,
        "bucket_distribution_json": serialize_json(bucket_distribution),
        "vip_distribution_json": serialize_json(vip_distribution),
        "score_by_segment_json": serialize_json(score_by_segment_rows),
        "eligible_population_json": serialize_json(eligible_population_rows),
        "alert_count": int(len(alerts)),
        "alerts_json": serialize_json(alerts),
        "pt": str(pt),
    }


def build_daily_markdown_report(
    *,
    pt: str,
    reference_pt: Optional[str],
    summary_row: Dict[str, Any],
    bucket_distribution: List[Dict[str, Any]],
    vip_distribution: List[Dict[str, Any]],
    eligible_population_rows: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
) -> List[str]:
    lines = [
        "# Phase-1 Response Daily Monitoring",
        "",
        f"- scoring partition: `{pt}`",
        f"- reference partition: `{reference_pt or 'none'}`",
        f"- monitor status: `{summary_row['monitor_status']}`",
        f"- row count: `{summary_row['row_count']}`",
        f"- distinct player count: `{summary_row['distinct_player_count']}`",
        f"- duplicate player count: `{summary_row['duplicate_player_count']}`",
        f"- invalid score count: `{summary_row['invalid_score_count']}`",
        f"- selected threshold: `{summary_row['selected_threshold']}`",
        f"- predicted positive users / rate: `{summary_row['positive_prediction_count']}` / `{summary_row['predicted_positive_rate']:.6f}`",
        f"- predicted negative users / rate: `{summary_row['negative_prediction_count']}` / `{summary_row['predicted_negative_rate']:.6f}`",
        f"- score mean / median: `{summary_row['score_mean']:.6f}` / `{summary_row['score_median']:.6f}`",
        f"- score min / max: `{summary_row['score_min']:.6f}` / `{summary_row['score_max']:.6f}`",
        f"- snapshot date span: `{summary_row['snapshot_date_min']}` to `{summary_row['snapshot_date_max']}`",
        f"- snapshot date lag days: `{summary_row['snapshot_date_lag_days']}`",
        "",
        "## Bucket Distribution",
        "",
        "| bucket | rows | share | avg_score | score_range |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in bucket_distribution:
        lines.append(
            f"| {row['bucket_name']} | {row['row_count']} | {row['row_share']:.4%} | {row['score_mean']:.6f} | {row['score_min']:.6f}..{row['score_max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## VIP Distribution",
            "",
            "| vip_level | rows | share | avg_score | median_score |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in vip_distribution:
        lines.append(
            f"| {row['segment_value']} | {row['row_count']} | {row['row_share']:.4%} | {row['avg_score']:.6f} | {row['median_score']:.6f} |"
        )
    if eligible_population_rows:
        lines.extend(
            [
                "",
                "## Eligible Population Summary",
                "",
                "| segment | eligible_rows | reference_rows | delta | delta_ratio | snapshot_range |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in eligible_population_rows:
            delta_ratio = "n/a" if row["eligible_row_delta_ratio"] is None else f"{row['eligible_row_delta_ratio']:.4%}"
            snapshot_range = f"{row['snapshot_date_min']}..{row['snapshot_date_max']}"
            lines.append(
                f"| {row['segment_value']} | {row['eligible_row_count']} | {row['reference_eligible_row_count']} | {row['eligible_row_delta']} | {delta_ratio} | {snapshot_range} |"
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
            "- This package monitors operational health and score behavior for the observational response scorer. It does not infer causal uplift or treatment effect.",
        ]
    )
    return lines
