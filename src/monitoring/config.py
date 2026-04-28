from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml_file(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SourceConfig:
    scored_table: str
    scored_partition_column: str
    feature_table: Optional[str]
    feature_partition_column: str
    eligibility_filter_sql: Optional[str]
    outcome_source_table: Optional[str]
    outcome_partition_column: str


@dataclass(frozen=True)
class SegmentConfig:
    score_segment_columns: List[str]
    daily_segment_columns: List[str]


@dataclass(frozen=True)
class DailyThresholdConfig:
    min_row_count: int
    row_count_drop_warn_ratio: float
    row_count_drop_alert_ratio: float
    row_count_growth_warn_ratio: float
    row_count_growth_alert_ratio: float
    duplicate_player_alert_count: int
    invalid_score_alert_count: int
    max_missing_metadata_rows: int
    max_snapshot_date_lag_days: int
    max_snapshot_date_distinct_count: int
    min_top_bucket_share: float
    max_top_bucket_share: float
    max_vip_missing_rate: float


@dataclass(frozen=True)
class PerformanceThresholdConfig:
    label_maturity_days: int
    response_window_days: int
    min_rows: int
    min_positive: int
    min_bucket_rows: int
    min_segment_rows: int
    response_rate_floor_warn: float
    response_rate_floor_alert: float
    calibration_gap_warn: float
    calibration_gap_alert: float


@dataclass(frozen=True)
class OutputConfig:
    reports_root: Path
    odps_daily_table: str
    odps_alerts_table: str
    odps_performance_table: str


@dataclass(frozen=True)
class ResponseMonitoringConfig:
    enabled: bool
    source: SourceConfig
    segments: SegmentConfig
    daily: DailyThresholdConfig
    performance: PerformanceThresholdConfig
    outputs: OutputConfig


def load_response_monitoring_config(path: Path) -> ResponseMonitoringConfig:
    raw = load_yaml_file(path)
    monitoring = dict(raw.get("response_monitoring", {}))
    source = dict(monitoring.get("source", {}))
    segments = dict(monitoring.get("segments", {}))
    daily = dict(monitoring.get("daily_thresholds", {}))
    performance = dict(monitoring.get("performance_thresholds", {}))
    outputs = dict(monitoring.get("outputs", {}))
    return ResponseMonitoringConfig(
        enabled=bool(monitoring.get("enabled", True)),
        source=SourceConfig(
            scored_table=str(source.get("scored_table", "pai_rec_prod.alg_uplift_phase1_response_scores_di")),
            scored_partition_column=str(source.get("scored_partition_column", "pt")),
            feature_table=None if source.get("feature_table") in {None, ""} else str(source.get("feature_table")),
            feature_partition_column=str(source.get("feature_partition_column", "pt")),
            eligibility_filter_sql=None if source.get("eligibility_filter_sql") in {None, ""} else str(source.get("eligibility_filter_sql")),
            outcome_source_table=None if source.get("outcome_source_table") in {None, ""} else str(source.get("outcome_source_table")),
            outcome_partition_column=str(source.get("outcome_partition_column", "pt")),
        ),
        segments=SegmentConfig(
            score_segment_columns=[str(value) for value in segments.get("score_segment_columns", ["vip_level"])],
            daily_segment_columns=[str(value) for value in segments.get("daily_segment_columns", ["vip_level"])],
        ),
        daily=DailyThresholdConfig(
            min_row_count=int(daily.get("min_row_count", 1)),
            row_count_drop_warn_ratio=float(daily.get("row_count_drop_warn_ratio", 0.10)),
            row_count_drop_alert_ratio=float(daily.get("row_count_drop_alert_ratio", 0.25)),
            row_count_growth_warn_ratio=float(daily.get("row_count_growth_warn_ratio", 0.15)),
            row_count_growth_alert_ratio=float(daily.get("row_count_growth_alert_ratio", 0.35)),
            duplicate_player_alert_count=int(daily.get("duplicate_player_alert_count", 1)),
            invalid_score_alert_count=int(daily.get("invalid_score_alert_count", 1)),
            max_missing_metadata_rows=int(daily.get("max_missing_metadata_rows", 0)),
            max_snapshot_date_lag_days=int(daily.get("max_snapshot_date_lag_days", 3)),
            max_snapshot_date_distinct_count=int(daily.get("max_snapshot_date_distinct_count", 2)),
            min_top_bucket_share=float(daily.get("min_top_bucket_share", 0.01)),
            max_top_bucket_share=float(daily.get("max_top_bucket_share", 0.70)),
            max_vip_missing_rate=float(daily.get("max_vip_missing_rate", 0.05)),
        ),
        performance=PerformanceThresholdConfig(
            label_maturity_days=int(performance.get("label_maturity_days", 4)),
            response_window_days=int(performance.get("response_window_days", 3)),
            min_rows=int(performance.get("min_rows", 1000)),
            min_positive=int(performance.get("min_positive", 25)),
            min_bucket_rows=int(performance.get("min_bucket_rows", 100)),
            min_segment_rows=int(performance.get("min_segment_rows", 100)),
            response_rate_floor_warn=float(performance.get("response_rate_floor_warn", 0.01)),
            response_rate_floor_alert=float(performance.get("response_rate_floor_alert", 0.005)),
            calibration_gap_warn=float(performance.get("calibration_gap_warn", 0.08)),
            calibration_gap_alert=float(performance.get("calibration_gap_alert", 0.15)),
        ),
        outputs=OutputConfig(
            reports_root=Path(str(outputs.get("reports_root", "reports/response_monitoring"))),
            odps_daily_table=str(outputs.get("odps_daily_table", "pai_rec_prod.ads_uplift_phase1_response_monitoring_daily_di")),
            odps_alerts_table=str(outputs.get("odps_alerts_table", "pai_rec_prod.ads_uplift_phase1_response_monitoring_alerts_di")),
            odps_performance_table=str(outputs.get("odps_performance_table", "pai_rec_prod.ads_uplift_phase1_response_monitoring_performance_di")),
        ),
    )
