from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from monitoring.config import ResponseMonitoringConfig, load_response_monitoring_config
    from monitoring.contracts import monitoring_table_contract
    from monitoring.daily_response_monitor import (
        build_bucket_distribution,
        build_daily_alerts,
        build_daily_markdown_report,
        build_daily_segment_summary_rows,
        build_daily_summary_row,
        build_population_summary,
        build_prediction_volume_summary,
        build_score_by_segment,
        build_segment_distribution,
        compute_score_distribution,
        required_scored_columns,
        validate_scored_frame,
    )
    from monitoring.odps_io import (
        alerts_to_frame,
        build_partition_exists_sql,
        ensure_table_reference,
        fetch_eligible_population_summary,
        fetch_observational_outcomes_for_scored_partition,
        fetch_partition_frame,
        fetch_scalar,
        fetch_table_column_names,
        list_partition_values,
        serialize_json,
        write_monitoring_frame_to_odps,
    )
    from monitoring.performance_response_monitor import (
        build_bucket_performance_rows,
        build_calibration_rows,
        build_global_performance_row,
        build_not_evaluable_global_row,
        build_performance_alerts,
        build_performance_markdown_report,
        build_segment_performance_rows,
        labels_are_mature_for_pt,
        validate_observational_frame,
    )
else:  # pragma: no cover
    from .config import ResponseMonitoringConfig, load_response_monitoring_config
    from .contracts import monitoring_table_contract
    from .daily_response_monitor import (
        build_bucket_distribution,
        build_daily_alerts,
        build_daily_markdown_report,
        build_daily_segment_summary_rows,
        build_daily_summary_row,
        build_population_summary,
        build_prediction_volume_summary,
        build_score_by_segment,
        build_segment_distribution,
        compute_score_distribution,
        required_scored_columns,
        validate_scored_frame,
    )
    from .odps_io import (
        alerts_to_frame,
        build_partition_exists_sql,
        ensure_table_reference,
        fetch_eligible_population_summary,
        fetch_observational_outcomes_for_scored_partition,
        fetch_partition_frame,
        fetch_scalar,
        fetch_table_column_names,
        list_partition_values,
        serialize_json,
        write_monitoring_frame_to_odps,
    )
    from .performance_response_monitor import (
        build_bucket_performance_rows,
        build_calibration_rows,
        build_global_performance_row,
        build_not_evaluable_global_row,
        build_performance_alerts,
        build_performance_markdown_report,
        build_segment_performance_rows,
        labels_are_mature_for_pt,
        validate_observational_frame,
    )

LOGGER = logging.getLogger("response.response_monitoring")


def bootstrap_runtime_environment() -> None:
    mpl_config_dir = (ROOT_DIR / ".mplconfig").resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)


def setup_logger(log_level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    return LOGGER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase-1 response monitoring for a single scored assignment date."
    )
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "configs" / "response_monitoring.yaml")
    parser.add_argument("--assignment_start_date", required=True, help="Assignment/scoring partition date in YYYYMMDD format.")
    parser.add_argument("--assignment_end_date", required=True, help="Assignment/scoring partition date in YYYYMMDD format.")
    parser.add_argument("--reference_pt", help="Optional explicit reference prediction partition for daily drift checks.")
    parser.add_argument("--run-label", help="Optional explicit monitoring run label.")
    parser.add_argument("--dry-run", action="store_true", help="Print monitoring plan and exit without ODPS writes.")
    parser.add_argument("--write-mode", default="overwrite", choices=["append", "overwrite"])
    parser.add_argument("--log-level", default="INFO")
    return parser


def resolve_run_label(explicit: Optional[str], *, current_pt: str) -> str:
    return explicit or f"response_monitoring_{current_pt}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"


def _single_value_or_none(series: pd.Series) -> Optional[Any]:
    values = series.dropna().astype(str).unique().tolist()
    if not values:
        return None
    if len(values) > 1:
        return ",".join(sorted(values))
    return values[0]


def _validate_assignment_dates(start_date: str, end_date: str) -> str:
    datetime.strptime(start_date, "%Y%m%d")
    datetime.strptime(end_date, "%Y%m%d")
    if start_date != end_date:
        raise ValueError(
            "Response monitoring requires assignment_start_date to equal assignment_end_date. "
            f"Received start={start_date} end={end_date}."
        )
    return end_date


def _resolve_reference_pt(config: ResponseMonitoringConfig, current_pt: str, explicit_reference_pt: Optional[str]) -> Optional[str]:
    if explicit_reference_pt:
        return explicit_reference_pt
    partitions = [partition for partition in list_partition_values(config.source.scored_table) if partition < current_pt]
    if not partitions:
        return None
    return partitions[-1]


def _fetch_prediction_partition_summary(config: ResponseMonitoringConfig, current_pt: str) -> Dict[str, Any]:
    partition_exists = int(
        fetch_scalar(
            build_partition_exists_sql(
                config.source.scored_table,
                config.source.scored_partition_column,
                current_pt,
            ),
            "row_count",
        )
        or 0
    )
    summary = {
        "partition_exists": partition_exists > 0,
        "row_count": partition_exists,
        "score_min": None,
        "score_max": None,
        "predicted_positive_count": None,
        "required_columns_present": None,
    }
    if partition_exists == 0:
        return summary

    scored = fetch_partition_frame(
        table_name=config.source.scored_table,
        partition_column=config.source.scored_partition_column,
        partition_value=current_pt,
        columns=required_scored_columns(),
    )
    validation_summary = validate_scored_frame(scored, expected_pt=current_pt)
    predicted_positive = build_prediction_volume_summary(
        scored,
        selected_threshold=float(pd.to_numeric(scored["selected_threshold"], errors="coerce").dropna().iloc[0])
        if scored["selected_threshold"].notna().any()
        else None,
    )
    score_distribution = compute_score_distribution(scored)
    summary.update(
        {
            "row_count": int(len(scored)),
            "score_min": score_distribution["score_min"],
            "score_max": score_distribution["score_max"],
            "predicted_positive_count": predicted_positive["positive_prediction_count"],
            "required_columns_present": True,
            "validation_summary": validation_summary,
        }
    )
    return summary


def _build_missing_partition_alert(config: ResponseMonitoringConfig, current_pt: str, reference_pt: Optional[str]) -> Dict[str, Any]:
    return {
        "monitor_run_ts": datetime.utcnow().replace(microsecond=0).isoformat(),
        "severity": "critical",
        "check_name": "missing_prediction_partition",
        "metric_name": "partition_present_flag",
        "observed_value": 0,
        "threshold_value": 1,
        "reference_value": reference_pt,
        "message": "Prediction partition is missing for the requested monitoring pt.",
        "context": {
            "prediction_table": config.source.scored_table,
            "partition_column": config.source.scored_partition_column,
            "current_pt": current_pt,
        },
    }


def _align_frame_to_table_schema(df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    target_columns = fetch_table_column_names(table_name)
    extra_columns = [column for column in df.columns if column not in target_columns]
    missing_columns = [column for column in target_columns if column not in df.columns]
    aligned = df.copy()
    for column in missing_columns:
        aligned[column] = None
    aligned = aligned.loc[:, target_columns]
    return aligned, {"extra_columns": extra_columns, "missing_columns": missing_columns}


def run_daily_monitoring(
    *,
    config: ResponseMonitoringConfig,
    current_pt: str,
    reference_pt: Optional[str],
    run_label: str,
) -> Dict[str, Any]:
    scored = fetch_partition_frame(
        table_name=config.source.scored_table,
        partition_column=config.source.scored_partition_column,
        partition_value=current_pt,
        columns=required_scored_columns(),
    )
    validation_summary = validate_scored_frame(scored, expected_pt=current_pt)

    reference_row_count = None
    if reference_pt:
        reference_scored = fetch_partition_frame(
            table_name=config.source.scored_table,
            partition_column=config.source.scored_partition_column,
            partition_value=reference_pt,
            columns=["player_id"],
        )
        reference_row_count = int(len(reference_scored))

    row_count = int(len(scored))
    distinct_player_count = int(scored["player_id"].astype("string").nunique())
    score_distribution = compute_score_distribution(scored)
    bucket_distribution = build_bucket_distribution(scored)
    vip_distribution = build_segment_distribution(scored, segment_column="vip_level")
    score_by_segment_rows = build_score_by_segment(scored, segment_column="vip_level")

    current_eligible = None
    reference_eligible = None
    if config.source.feature_table and config.source.eligibility_filter_sql:
        current_eligible = fetch_eligible_population_summary(
            table_name=config.source.feature_table,
            partition_column=config.source.feature_partition_column,
            partition_value=current_pt,
            eligibility_filter_sql=config.source.eligibility_filter_sql,
            segment_column="vip_level",
        )
        if reference_pt:
            reference_eligible = fetch_eligible_population_summary(
                table_name=config.source.feature_table,
                partition_column=config.source.feature_partition_column,
                partition_value=reference_pt,
                eligibility_filter_sql=config.source.eligibility_filter_sql,
                segment_column="vip_level",
            )
    eligible_population_rows = build_population_summary(
        current_eligible=current_eligible,
        reference_eligible=reference_eligible,
        segment_column="vip_level",
    )

    metadata_missing_rows = int(
        scored[["model_name", "model_version", "model_reference_path", "selected_threshold"]].isna().any(axis=1).sum()
    )
    selected_threshold = (
        float(pd.to_numeric(scored["selected_threshold"], errors="coerce").dropna().iloc[0])
        if scored["selected_threshold"].notna().any()
        else None
    )
    prediction_volume_summary = build_prediction_volume_summary(scored, selected_threshold=selected_threshold)
    snapshot_dates = pd.to_datetime(scored["snapshot_date"], errors="coerce")
    score_dates = pd.to_datetime(scored["score_date"], errors="coerce")
    snapshot_date_min = snapshot_dates.min().date().isoformat() if snapshot_dates.notna().any() else None
    snapshot_date_max = snapshot_dates.max().date().isoformat() if snapshot_dates.notna().any() else None
    snapshot_date_distinct_count = int(snapshot_dates.dt.date.nunique()) if snapshot_dates.notna().any() else 0
    snapshot_date_lag_days = None
    if snapshot_dates.notna().any() and score_dates.notna().any():
        snapshot_date_lag_days = int((score_dates.max().date() - snapshot_dates.max().date()).days)

    alerts = build_daily_alerts(
        config=config,
        row_count=row_count,
        reference_row_count=reference_row_count,
        validation_summary=validation_summary,
        score_distribution=score_distribution,
        bucket_distribution=bucket_distribution,
        vip_distribution=vip_distribution,
        metadata_missing_rows=metadata_missing_rows,
        snapshot_date_lag_days=snapshot_date_lag_days,
        snapshot_date_distinct_count=snapshot_date_distinct_count,
    )

    summary_row = build_daily_summary_row(
        config=config,
        pt=current_pt,
        reference_pt=reference_pt,
        row_count=row_count,
        distinct_player_count=distinct_player_count,
        validation_summary=validation_summary,
        score_distribution=score_distribution,
        bucket_distribution=bucket_distribution,
        vip_distribution=vip_distribution,
        score_by_segment_rows=score_by_segment_rows,
        eligible_population_rows=eligible_population_rows,
        metadata_missing_rows=metadata_missing_rows,
        model_name=_single_value_or_none(scored["model_name"]),
        model_version=_single_value_or_none(scored["model_version"]),
        model_reference_path=_single_value_or_none(scored["model_reference_path"]),
        selected_threshold=selected_threshold,
        prediction_volume_summary=prediction_volume_summary,
        snapshot_date_min=snapshot_date_min,
        snapshot_date_max=snapshot_date_max,
        snapshot_date_distinct_count=snapshot_date_distinct_count,
        snapshot_date_lag_days=snapshot_date_lag_days,
        partition_present_flag=1,
        alerts=alerts,
    )
    summary_row["run_label"] = run_label
    segment_summary_rows = build_daily_segment_summary_rows(
        config=config,
        scored=scored,
        pt=current_pt,
        reference_pt=reference_pt,
        segment_columns=config.segments.daily_segment_columns,
        model_name=summary_row["model_name"],
        model_version=summary_row["model_version"],
        model_reference_path=summary_row["model_reference_path"],
        selected_threshold=summary_row["selected_threshold"],
    )
    for row in segment_summary_rows:
        row["run_label"] = run_label

    markdown_lines = build_daily_markdown_report(
        pt=current_pt,
        reference_pt=reference_pt,
        summary_row=summary_row,
        bucket_distribution=bucket_distribution,
        vip_distribution=vip_distribution,
        eligible_population_rows=eligible_population_rows,
        alerts=alerts,
    )
    return {
        "run_label": run_label,
        "mode": "daily",
        "summary_rows": [summary_row] + segment_summary_rows,
        "alerts": alerts,
        "markdown_lines": markdown_lines,
    }


def run_performance_monitoring(
    *,
    config: ResponseMonitoringConfig,
    current_pt: str,
    run_label: str,
) -> Dict[str, Any]:
    maturity_summary = labels_are_mature_for_pt(
        current_pt,
        as_of_date=date.today(),
        label_maturity_days=config.performance.label_maturity_days,
        response_window_days=config.performance.response_window_days,
    )
    if not bool(maturity_summary["labels_mature"]):
        global_row = build_not_evaluable_global_row(pt=current_pt, status_reason="labels_not_mature")
        global_row["run_label"] = run_label
        markdown_lines = build_performance_markdown_report(
            pt=current_pt,
            maturity_summary=maturity_summary,
            global_row=global_row,
            bucket_rows=[],
            segment_rows=[],
            calibration_rows=[],
            alerts=[],
        )
        return {
            "run_label": run_label,
            "mode": "performance",
            "summary_rows": [global_row],
            "bucket_rows": [],
            "segment_rows": [],
            "calibration_rows": [],
            "alerts": [],
            "markdown_lines": markdown_lines,
            "maturity_summary": maturity_summary,
        }

    if not config.source.outcome_source_table:
        raise ValueError("Delayed response monitoring requires `outcome_source_table` in config.")
    observational_frame = fetch_observational_outcomes_for_scored_partition(
        scored_table=config.source.scored_table,
        scored_partition_column=config.source.scored_partition_column,
        scoring_pt=current_pt,
        outcome_source_table=config.source.outcome_source_table,
        outcome_partition_column=config.source.outcome_partition_column,
        response_window_days=config.performance.response_window_days,
    )
    validate_observational_frame(observational_frame, response_window_days=config.performance.response_window_days)

    global_row = build_global_performance_row(
        observational_frame,
        pt=current_pt,
        response_window_days=config.performance.response_window_days,
    )
    global_row["run_label"] = run_label
    bucket_rows = build_bucket_performance_rows(
        observational_frame,
        pt=current_pt,
        response_window_days=config.performance.response_window_days,
        min_rows=config.performance.min_bucket_rows,
    )
    for row in bucket_rows:
        row["run_label"] = run_label
    segment_rows = build_segment_performance_rows(
        observational_frame,
        pt=current_pt,
        response_window_days=config.performance.response_window_days,
        segment_columns=config.segments.score_segment_columns,
        min_rows=config.performance.min_segment_rows,
    )
    for row in segment_rows:
        row["run_label"] = run_label
    calibration_rows = build_calibration_rows(
        observational_frame,
        pt=current_pt,
        response_window_days=config.performance.response_window_days,
    )
    alerts = build_performance_alerts(
        config=config,
        global_row=global_row,
        calibration_rows=calibration_rows,
    )
    markdown_lines = build_performance_markdown_report(
        pt=current_pt,
        maturity_summary=maturity_summary,
        global_row=global_row,
        bucket_rows=bucket_rows,
        segment_rows=segment_rows,
        calibration_rows=calibration_rows,
        alerts=alerts,
    )
    return {
        "run_label": run_label,
        "mode": "performance",
        "summary_rows": [global_row],
        "bucket_rows": bucket_rows,
        "segment_rows": segment_rows,
        "calibration_rows": calibration_rows,
        "alerts": alerts,
        "markdown_lines": markdown_lines,
        "maturity_summary": maturity_summary,
    }


def maybe_write_alert_before_failure(
    *,
    config: ResponseMonitoringConfig,
    current_pt: str,
    run_label: str,
    alert: Dict[str, Any],
    write_mode: str,
) -> None:
    alert_frame = alerts_to_frame([alert], pt=current_pt, mode="daily", run_label=run_label)
    aligned_frame, mismatch = _align_frame_to_table_schema(alert_frame, ensure_table_reference(config.outputs.odps_alerts_table))
    if mismatch["extra_columns"] or mismatch["missing_columns"]:
        LOGGER.warning(
            "alerts_table_schema_alignment extra=%s missing=%s",
            mismatch["extra_columns"],
            mismatch["missing_columns"],
        )
    write_monitoring_frame_to_odps(
        aligned_frame,
        odps_target=ensure_table_reference(config.outputs.odps_alerts_table),
        partition_column="pt",
        partition_value=current_pt,
        write_mode=write_mode,
    )


def maybe_write_monitoring_results(
    *,
    config: ResponseMonitoringConfig,
    current_pt: str,
    daily_result: Dict[str, Any],
    performance_result: Dict[str, Any],
    write_mode: str,
) -> Dict[str, Dict[str, List[str]]]:
    tables = monitoring_table_contract(
        {
            "odps_daily_table": config.outputs.odps_daily_table,
            "odps_performance_table": config.outputs.odps_performance_table,
            "odps_alerts_table": config.outputs.odps_alerts_table,
        }
    )
    schema_mismatches: Dict[str, Dict[str, List[str]]] = {}

    daily_frame, daily_mismatch = _align_frame_to_table_schema(
        pd.DataFrame(daily_result["summary_rows"]),
        ensure_table_reference(tables["daily"]),
    )
    schema_mismatches["daily"] = daily_mismatch
    write_monitoring_frame_to_odps(
        daily_frame,
        odps_target=ensure_table_reference(tables["daily"]),
        partition_column="pt",
        partition_value=current_pt,
        write_mode=write_mode,
    )

    performance_rows = (
        list(performance_result["summary_rows"])
        + list(performance_result.get("bucket_rows", []))
        + list(performance_result.get("segment_rows", []))
    )
    performance_frame, performance_mismatch = _align_frame_to_table_schema(
        pd.DataFrame(performance_rows),
        ensure_table_reference(tables["performance"]),
    )
    schema_mismatches["performance"] = performance_mismatch
    write_monitoring_frame_to_odps(
        performance_frame,
        odps_target=ensure_table_reference(tables["performance"]),
        partition_column="pt",
        partition_value=current_pt,
        write_mode=write_mode,
    )

    alert_frame = pd.concat(
        [
            alerts_to_frame(
                list(daily_result["alerts"]),
                pt=current_pt,
                mode="daily",
                run_label=daily_result["run_label"],
            ),
            alerts_to_frame(
                list(performance_result["alerts"]),
                pt=current_pt,
                mode="performance",
                run_label=performance_result["run_label"],
            ),
        ],
        ignore_index=True,
    )
    aligned_alert_frame, alerts_mismatch = _align_frame_to_table_schema(
        alert_frame,
        ensure_table_reference(tables["alerts"]),
    )
    schema_mismatches["alerts"] = alerts_mismatch
    write_monitoring_frame_to_odps(
        aligned_alert_frame,
        odps_target=ensure_table_reference(tables["alerts"]),
        partition_column="pt",
        partition_value=current_pt,
        write_mode=write_mode,
    )
    return schema_mismatches


def print_dry_run_plan(
    *,
    config: ResponseMonitoringConfig,
    current_pt: str,
    reference_pt: Optional[str],
    prediction_summary: Dict[str, Any],
    maturity_summary: Dict[str, Any],
) -> None:
    lines = [
        f"[DRY_RUN] current_pt={current_pt}",
        f"[DRY_RUN] reference_pt={reference_pt or 'none'}",
        f"[DRY_RUN] prediction_table={config.source.scored_table}",
        f"[DRY_RUN] monitoring_daily_table={config.outputs.odps_daily_table}",
        f"[DRY_RUN] monitoring_performance_table={config.outputs.odps_performance_table}",
        f"[DRY_RUN] monitoring_alerts_table={config.outputs.odps_alerts_table}",
        f"[DRY_RUN] current_prediction_partition_exists={prediction_summary['partition_exists']}",
        f"[DRY_RUN] current_prediction_partition_row_count={prediction_summary['row_count']}",
        f"[DRY_RUN] current_prediction_score_min={prediction_summary['score_min']}",
        f"[DRY_RUN] current_prediction_score_max={prediction_summary['score_max']}",
        f"[DRY_RUN] current_prediction_predicted_positive_count={prediction_summary['predicted_positive_count']}",
        f"[DRY_RUN] required_columns_present={prediction_summary['required_columns_present']}",
        f"[DRY_RUN] labels_mature={maturity_summary['labels_mature']}",
        f"[DRY_RUN] matured_on_date={maturity_summary['matured_on_date']}",
    ]
    if prediction_summary["partition_exists"] is True:
        lines.append("[DRY_RUN] planned_writes=daily_monitoring,performance_monitoring,alerts")
    elif prediction_summary["partition_exists"] is False:
        lines.append("[DRY_RUN] planned_writes=alerts_only_then_fail")
    else:
        lines.append("[DRY_RUN] planned_writes=unknown_without_live_odps_access")
    for line in lines:
        print(line, flush=True)


def main() -> None:
    bootstrap_runtime_environment()
    args = build_parser().parse_args()
    setup_logger(args.log_level)

    current_pt = _validate_assignment_dates(args.assignment_start_date, args.assignment_end_date)
    config = load_response_monitoring_config(args.config.resolve())
    run_label = resolve_run_label(args.run_label, current_pt=current_pt)
    maturity_summary = labels_are_mature_for_pt(
        current_pt,
        as_of_date=date.today(),
        label_maturity_days=config.performance.label_maturity_days,
        response_window_days=config.performance.response_window_days,
    )
    reference_pt = args.reference_pt
    prediction_summary: Dict[str, Any] = {
        "partition_exists": None,
        "row_count": None,
        "score_min": None,
        "score_max": None,
        "predicted_positive_count": None,
        "required_columns_present": None,
    }
    live_query_error: Optional[Exception] = None
    try:
        reference_pt = _resolve_reference_pt(config, current_pt, args.reference_pt)
        prediction_summary = _fetch_prediction_partition_summary(config, current_pt)
    except Exception as exc:
        live_query_error = exc

    if args.dry_run:
        print_dry_run_plan(
            config=config,
            current_pt=current_pt,
            reference_pt=reference_pt,
            prediction_summary=prediction_summary,
            maturity_summary=maturity_summary,
        )
        if live_query_error is not None:
            print(f"[DRY_RUN] live_odps_check_unavailable={live_query_error}", flush=True)
        return

    if live_query_error is not None:
        raise live_query_error

    if not prediction_summary["partition_exists"]:
        alert = _build_missing_partition_alert(config, current_pt, reference_pt)
        maybe_write_alert_before_failure(
            config=config,
            current_pt=current_pt,
            run_label=run_label,
            alert=alert,
            write_mode=args.write_mode,
        )
        raise RuntimeError(
            f"Prediction partition is missing in {config.source.scored_table} for pt={current_pt}. "
            "Alert was written before aborting monitoring."
        )

    daily_result = run_daily_monitoring(
        config=config,
        current_pt=current_pt,
        reference_pt=reference_pt,
        run_label=run_label,
    )
    performance_result = run_performance_monitoring(
        config=config,
        current_pt=current_pt,
        run_label=run_label,
    )
    schema_mismatches = maybe_write_monitoring_results(
        config=config,
        current_pt=current_pt,
        daily_result=daily_result,
        performance_result=performance_result,
        write_mode=args.write_mode,
    )
    LOGGER.info(
        "response_monitoring_complete pt=%s reference_pt=%s labels_mature=%s schema_mismatches=%s",
        current_pt,
        reference_pt,
        maturity_summary["labels_mature"],
        serialize_json(schema_mismatches),
    )


if __name__ == "__main__":
    main()
