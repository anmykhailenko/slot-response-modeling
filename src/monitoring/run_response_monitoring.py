from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from monitoring.config import load_response_monitoring_config
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
        write_monitoring_frame_to_odps,
    )
    from monitoring.performance_response_monitor import (
        build_bucket_performance_rows,
        build_calibration_rows,
        build_global_performance_row,
        build_performance_alerts,
        build_performance_markdown_report,
        build_segment_performance_rows,
        mature_outcome_window_for_pt,
        validate_observational_frame,
    )
else:  # pragma: no cover
    from .config import load_response_monitoring_config
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
        write_monitoring_frame_to_odps,
    )
    from .performance_response_monitor import (
        build_bucket_performance_rows,
        build_calibration_rows,
        build_global_performance_row,
        build_performance_alerts,
        build_performance_markdown_report,
        build_segment_performance_rows,
        mature_outcome_window_for_pt,
        validate_observational_frame,
    )
LOGGER = logging.getLogger("response.response_monitoring")


def bootstrap_runtime_environment() -> None:
    mpl_config_dir = (PROJECT_ROOT / ".mplconfig").resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)


def setup_logger(log_level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    return LOGGER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run production monitoring for the Phase-1 observational response scoring pipeline.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "response_monitoring.yaml")
    parser.add_argument("--mode", choices=["daily", "performance"], required=True)
    parser.add_argument("--pt", required=True, help="Scoring partition date in YYYYMMDD format.")
    parser.add_argument("--reference-pt", help="Reference partition for daily response monitoring.")
    parser.add_argument("--run-label", help="Optional explicit monitoring run label.")
    parser.add_argument("--write-odps", action="store_true", help="Write monitoring outputs to configured ODPS monitoring tables.")
    parser.add_argument("--write-mode", default="overwrite", choices=["append", "overwrite"], help="ODPS write mode for monitoring outputs.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def resolve_run_label(explicit: Optional[str], *, pt: str, mode: str) -> str:
    return explicit or f"{mode}_{pt}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"


def _single_value_or_none(series: pd.Series) -> Optional[Any]:
    values = series.dropna().astype(str).unique().tolist()
    if not values:
        return None
    if len(values) > 1:
        return ",".join(sorted(values))
    return values[0]


def run_daily_monitoring(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_response_monitoring_config(args.config.resolve())
    run_label = resolve_run_label(args.run_label, pt=args.pt, mode="daily")
    LOGGER.info("response_daily_monitoring_start pt=%s reference_pt=%s run_label=%s", args.pt, args.reference_pt, run_label)

    scored = fetch_partition_frame(
        table_name=config.source.scored_table,
        partition_column=config.source.scored_partition_column,
        partition_value=args.pt,
        columns=required_scored_columns(),
    )
    partition_present_count = fetch_scalar(
        build_partition_exists_sql(config.source.scored_table, config.source.scored_partition_column, args.pt),
        "row_count",
    )
    partition_present_flag = 1 if int(partition_present_count or 0) > 0 else 0
    validation_summary = validate_scored_frame(scored, expected_pt=args.pt)

    reference_row_count = None
    if args.reference_pt:
        reference_scored = fetch_partition_frame(
            table_name=config.source.scored_table,
            partition_column=config.source.scored_partition_column,
            partition_value=args.reference_pt,
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
            partition_value=args.pt,
            eligibility_filter_sql=config.source.eligibility_filter_sql,
            segment_column="vip_level",
        )
        if args.reference_pt:
            reference_eligible = fetch_eligible_population_summary(
                table_name=config.source.feature_table,
                partition_column=config.source.feature_partition_column,
                partition_value=args.reference_pt,
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
    selected_threshold = float(pd.to_numeric(scored["selected_threshold"], errors="coerce").dropna().iloc[0]) if scored["selected_threshold"].notna().any() else None
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
        pt=args.pt,
        reference_pt=args.reference_pt,
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
        partition_present_flag=partition_present_flag,
        alerts=alerts,
    )
    summary_row["run_label"] = run_label
    segment_summary_rows = build_daily_segment_summary_rows(
        config=config,
        scored=scored,
        pt=args.pt,
        reference_pt=args.reference_pt,
        segment_columns=config.segments.daily_segment_columns,
        model_name=summary_row["model_name"],
        model_version=summary_row["model_version"],
        model_reference_path=summary_row["model_reference_path"],
        selected_threshold=summary_row["selected_threshold"],
    )
    for row in segment_summary_rows:
        row["run_label"] = run_label

    markdown_lines = build_daily_markdown_report(
        pt=args.pt,
        reference_pt=args.reference_pt,
        summary_row=summary_row,
        bucket_distribution=bucket_distribution,
        vip_distribution=vip_distribution,
        eligible_population_rows=eligible_population_rows,
        alerts=alerts,
    )
    artifact_payloads = {
        "daily_summary.csv": [summary_row],
        "bucket_distribution.csv": bucket_distribution,
        "vip_distribution.csv": vip_distribution,
        "score_by_segment.csv": score_by_segment_rows,
        "eligible_population_summary.csv": eligible_population_rows,
        "alerts.json": {"alerts": alerts},
        "daily_response_monitoring_report.md": markdown_lines,
    }
    result = {
        "run_label": run_label,
        "mode": "daily",
        "summary_rows": [summary_row] + segment_summary_rows,
        "alerts": alerts,
        "artifact_payloads": artifact_payloads,
        "output_dir": None,
    }
    LOGGER.info("response_daily_monitoring_complete pt=%s alerts=%s", args.pt, len(alerts))
    return result


def run_performance_monitoring(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_response_monitoring_config(args.config.resolve())
    run_label = resolve_run_label(args.run_label, pt=args.pt, mode="performance")
    LOGGER.info("response_performance_monitoring_start pt=%s run_label=%s", args.pt, run_label)

    if not config.source.outcome_source_table:
        raise ValueError("Delayed response monitoring requires `outcome_source_table` in config.")
    observational_frame = fetch_observational_outcomes_for_scored_partition(
        scored_table=config.source.scored_table,
        scored_partition_column=config.source.scored_partition_column,
        scoring_pt=args.pt,
        outcome_source_table=config.source.outcome_source_table,
        outcome_partition_column=config.source.outcome_partition_column,
        response_window_days=config.performance.response_window_days,
    )
    validate_observational_frame(observational_frame, response_window_days=config.performance.response_window_days)

    maturity_summary = mature_outcome_window_for_pt(
        args.pt,
        label_maturity_days=config.performance.label_maturity_days,
        response_window_days=config.performance.response_window_days,
    )
    global_row = build_global_performance_row(
        observational_frame,
        pt=args.pt,
        response_window_days=config.performance.response_window_days,
    )
    global_row["run_label"] = run_label
    bucket_rows = build_bucket_performance_rows(
        observational_frame,
        pt=args.pt,
        response_window_days=config.performance.response_window_days,
        min_rows=config.performance.min_bucket_rows,
    )
    for row in bucket_rows:
        row["run_label"] = run_label
    segment_rows = build_segment_performance_rows(
        observational_frame,
        pt=args.pt,
        response_window_days=config.performance.response_window_days,
        segment_columns=config.segments.score_segment_columns,
        min_rows=config.performance.min_segment_rows,
    )
    for row in segment_rows:
        row["run_label"] = run_label
    calibration_rows = build_calibration_rows(
        observational_frame,
        pt=args.pt,
        response_window_days=config.performance.response_window_days,
    )
    alerts = build_performance_alerts(
        config=config,
        global_row=global_row,
        calibration_rows=calibration_rows,
    )
    markdown_lines = build_performance_markdown_report(
        pt=args.pt,
        maturity_summary=maturity_summary,
        global_row=global_row,
        bucket_rows=bucket_rows,
        segment_rows=segment_rows,
        calibration_rows=calibration_rows,
        alerts=alerts,
    )
    artifact_payloads = {
        "performance_summary.csv": [global_row],
        "bucket_observational_performance.csv": bucket_rows,
        "segment_observational_performance.csv": segment_rows,
        "calibration_summary.csv": calibration_rows,
        "alerts.json": {"alerts": alerts},
        "performance_response_monitoring_report.md": markdown_lines,
    }
    result = {
        "run_label": run_label,
        "mode": "performance",
        "summary_rows": [global_row],
        "bucket_rows": bucket_rows,
        "segment_rows": segment_rows,
        "calibration_rows": calibration_rows,
        "alerts": alerts,
        "artifact_payloads": artifact_payloads,
        "output_dir": None,
    }
    LOGGER.info("response_performance_monitoring_complete pt=%s alerts=%s", args.pt, len(alerts))
    return result


def maybe_write_monitoring_results_to_odps(args: argparse.Namespace, result: Dict[str, Any]) -> None:
    if not args.write_odps:
        return
    config = load_response_monitoring_config(args.config.resolve())
    tables = monitoring_table_contract(
        {
            "odps_daily_table": config.outputs.odps_daily_table,
            "odps_performance_table": config.outputs.odps_performance_table,
            "odps_alerts_table": config.outputs.odps_alerts_table,
        }
    )
    if result["mode"] == "daily":
        write_monitoring_frame_to_odps(
            pd.DataFrame(result["summary_rows"]),
            odps_target=ensure_table_reference(tables["daily"]),
            partition_column="pt",
            partition_value=args.pt,
            write_mode=args.write_mode,
        )
    elif result["mode"] == "performance":
        performance_rows = list(result["summary_rows"]) + list(result.get("bucket_rows", [])) + list(result.get("segment_rows", []))
        write_monitoring_frame_to_odps(
            pd.DataFrame(performance_rows),
            odps_target=ensure_table_reference(tables["performance"]),
            partition_column="pt",
            partition_value=args.pt,
            write_mode=args.write_mode,
        )
    alert_frame = alerts_to_frame(result["alerts"], pt=args.pt, mode=result["mode"], run_label=result["run_label"])
    write_monitoring_frame_to_odps(
        alert_frame,
        odps_target=ensure_table_reference(tables["alerts"]),
        partition_column="pt",
        partition_value=args.pt,
        write_mode=args.write_mode,
    )


def main() -> None:
    bootstrap_runtime_environment()
    args = build_parser().parse_args()
    setup_logger(args.log_level)
    if args.mode == "daily" and not args.reference_pt:
        LOGGER.warning("Daily response monitoring is running without a reference partition; row-count drift checks will be limited.")
    if args.mode == "daily":
        result = run_daily_monitoring(args)
    else:
        result = run_performance_monitoring(args)
    maybe_write_monitoring_results_to_odps(args, result)
    LOGGER.info("response_monitoring_result=%s", {"mode": result["mode"], "run_label": result["run_label"], "output_dir": result["output_dir"]})


if __name__ == "__main__":
    main()
