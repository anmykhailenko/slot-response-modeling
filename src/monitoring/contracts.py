from __future__ import annotations

from typing import Dict, List

DETAILS_DIRNAME = "details"

DAILY_ARTIFACT_FILENAMES: List[str] = [
    "daily_summary.csv",
    "bucket_distribution.csv",
    "vip_distribution.csv",
    "score_by_segment.csv",
    "eligible_population_summary.csv",
    "alerts.json",
    "daily_response_monitoring_report.md",
]

PERFORMANCE_ARTIFACT_FILENAMES: List[str] = [
    "performance_summary.csv",
    "bucket_observational_performance.csv",
    "segment_observational_performance.csv",
    "calibration_summary.csv",
    "alerts.json",
    "performance_response_monitoring_report.md",
]


def monitoring_artifact_contract(mode: str) -> List[str]:
    normalized = str(mode).strip().lower()
    if normalized == "daily":
        return list(DAILY_ARTIFACT_FILENAMES)
    if normalized == "performance":
        return list(PERFORMANCE_ARTIFACT_FILENAMES)
    raise ValueError(f"Unsupported response monitoring mode: {mode}")


def monitoring_table_contract(outputs: Dict[str, str]) -> Dict[str, str]:
    return {
        "daily": str(outputs["odps_daily_table"]),
        "performance": str(outputs["odps_performance_table"]),
        "alerts": str(outputs["odps_alerts_table"]),
    }
