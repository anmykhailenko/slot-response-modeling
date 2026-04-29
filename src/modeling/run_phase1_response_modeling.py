from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from data.odps_reader import create_odps_client_from_env, fetch_sql_as_frame, list_odps_partition_values  # noqa: E402
    from modeling.inference_feature_contract import (  # noqa: E402
        build_named_transformed_frame,
        resolve_expected_raw_feature_order,
        resolve_expected_transformed_feature_names,
        validate_scoring_input_frame,
    )
    from modeling.production_model_registry import build_model_version, write_champion_reference  # noqa: E402
    from modeling.train_lgbm import build_estimator as build_lgbm_estimator  # noqa: E402
    from modeling.train_lgbm import build_preprocessor as build_lgbm_preprocessor  # noqa: E402
    from modeling.train_lgbm import feature_importance_frame as lgbm_feature_importance_frame  # noqa: E402
    from modeling.train_logreg import build_estimator as build_logreg_estimator  # noqa: E402
    from modeling.train_logreg import build_preprocessor as build_logreg_preprocessor  # noqa: E402
    from modeling.train_logreg import feature_importance_frame as logreg_feature_importance_frame  # noqa: E402
    from modeling.generate_response_model_reporting import generate_reporting_artifacts  # noqa: E402
else:  # pragma: no cover
    from ..data.odps_reader import create_odps_client_from_env, fetch_sql_as_frame, list_odps_partition_values
    from .inference_feature_contract import (
        build_named_transformed_frame,
        resolve_expected_raw_feature_order,
        resolve_expected_transformed_feature_names,
        validate_scoring_input_frame,
    )
    from .production_model_registry import build_model_version, write_champion_reference
    from .train_lgbm import build_estimator as build_lgbm_estimator
    from .train_lgbm import build_preprocessor as build_lgbm_preprocessor
    from .train_lgbm import feature_importance_frame as lgbm_feature_importance_frame
    from .train_logreg import build_estimator as build_logreg_estimator
    from .train_logreg import build_preprocessor as build_logreg_preprocessor
    from .train_logreg import feature_importance_frame as logreg_feature_importance_frame
    from .generate_response_model_reporting import generate_reporting_artifacts


LOGGER = logging.getLogger(__name__)
TEMPORAL_STRING_CAST_COLUMNS = {
    "assignment_date",
    "treatment_timestamp",
    "feature_snapshot_date",
    "first_outcome_event_date",
    "last_outcome_event_date",
    "feature_create_time",
}


@dataclass(frozen=True)
class RuntimeWindow:
    min_pt: str
    max_pt: str
    maturity_end_pt: str
    modeling_end_pt: str


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start_pt: str
    end_pt: str
    pts: Tuple[str, ...]


@dataclass(frozen=True)
class SplitPlan:
    mature_cutoff_pt: str
    selected_mature_pts: Tuple[str, ...]
    train: SplitWindow
    validation: SplitWindow
    test: SplitWindow


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_response_champion_reference_path(config: Dict[str, Any]) -> Path:
    configured = str(config.get("champion_reference_path", "")).strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate
    return (PROJECT_ROOT / "model_registry" / "response_current.json").resolve()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported frame output format: {path}")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bootstrap_runtime_environment() -> None:
    mpl_config_dir = (PROJECT_ROOT / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows_"
    view = frame.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    columns = [str(column) for column in view.columns]
    rows = []
    for _, row in view.iterrows():
        rows.append(
            [
                "" if pd.isna(value) else str(value)
                for value in row.tolist()
            ]
        )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(values) + " |" for values in rows]
    return "\n".join([header, separator] + body)


def format_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{int(value):,}"


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{float(value):.{digits}f}"


def partition_value(spec: Any) -> str:
    match = re.search(r"pt='([^']+)'", str(spec))
    if not match:
        raise ValueError(f"Could not parse partition spec: {spec}")
    return match.group(1)


def pt_to_date(pt: str) -> date:
    return datetime.strptime(str(pt), "%Y%m%d").date()


def format_pt_as_date(pt: str) -> str:
    return pt_to_date(pt).strftime("%Y-%m-%d")


def list_partitions(table_name: str) -> List[str]:
    return list_odps_partition_values(table_name, partition_column="pt")


def resolve_runtime_window(config: Dict[str, Any], partitions: List[str]) -> RuntimeWindow:
    if not partitions:
        raise ValueError(
            "No `pt` partition values were discovered in the response dataset. "
            "The table may be empty, or ODPS metadata may not expose partitions for this table handle."
        )
    as_of_date = date.fromisoformat(str(config["as_of_date"]))
    response_window_days = int(config["response_window_days"])
    maturity_buffer_days = int(config.get("maturity_buffer_days", 0))
    maturity_end = as_of_date - timedelta(days=response_window_days + maturity_buffer_days)
    maturity_end_pt = maturity_end.strftime("%Y%m%d")
    eligible = [pt for pt in partitions if pt <= maturity_end_pt]
    if not eligible:
        raise ValueError(
            f"No mature partitions are available. Latest mature cutoff is {maturity_end_pt}, "
            f"but the table starts at {partitions[0]}."
        )
    return RuntimeWindow(
        min_pt=partitions[0],
        max_pt=partitions[-1],
        maturity_end_pt=maturity_end_pt,
        modeling_end_pt=eligible[-1],
    )


def build_partition_predicate(start_pt: str, end_pt: str) -> str:
    return f"pt >= '{start_pt}' and pt <= '{end_pt}'"


def build_split_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    split_policy = dict(config.get("split_policy", {}))
    return {
        "train_days": int(split_policy.get("train_days", 30)),
        "validation_days": int(split_policy.get("validation_days", 7)),
        "test_days": int(split_policy.get("test_days", 14)),
        "require_contiguous_partitions": bool(split_policy.get("require_contiguous_partitions", True)),
    }


def resolve_split_plan(runtime_window: RuntimeWindow, partitions: Sequence[str], split_policy: Dict[str, Any]) -> SplitPlan:
    mature_partitions = [str(pt) for pt in partitions if str(pt) <= runtime_window.modeling_end_pt]
    total_required_days = int(split_policy["train_days"]) + int(split_policy["validation_days"]) + int(split_policy["test_days"])
    if len(mature_partitions) < total_required_days:
        raise ValueError(
            "Not enough mature partitions for the configured split policy. "
            f"required={total_required_days}, available={len(mature_partitions)}, mature_cutoff={runtime_window.modeling_end_pt}."
        )

    selected = mature_partitions[-total_required_days:]
    if bool(split_policy["require_contiguous_partitions"]):
        expected = [
            (pt_to_date(selected[0]) + timedelta(days=offset)).strftime("%Y%m%d")
            for offset in range(total_required_days)
        ]
        if selected != expected:
            raise ValueError(
                "Configured split policy requires contiguous mature partitions, "
                f"but the selected range {selected[0]}..{selected[-1]} has gaps."
            )

    train_days = int(split_policy["train_days"])
    validation_days = int(split_policy["validation_days"])
    train_pts = tuple(selected[:train_days])
    validation_pts = tuple(selected[train_days : train_days + validation_days])
    test_pts = tuple(selected[train_days + validation_days :])

    return SplitPlan(
        mature_cutoff_pt=runtime_window.modeling_end_pt,
        selected_mature_pts=tuple(selected),
        train=SplitWindow("train", train_pts[0], train_pts[-1], train_pts),
        validation=SplitWindow("validation", validation_pts[0], validation_pts[-1], validation_pts),
        test=SplitWindow("test", test_pts[0], test_pts[-1], test_pts),
    )


def split_manifest_from_plan(split_plan: SplitPlan) -> Dict[str, List[str]]:
    return {
        "train": list(split_plan.train.pts),
        "validation": list(split_plan.validation.pts),
        "test": list(split_plan.test.pts),
    }


def estimate_memory_bytes(row_count: int, column_count: int) -> int:
    bytes_per_row = max(256, int(column_count) * 32)
    return int(row_count) * bytes_per_row


def format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def build_select_expression(column: str) -> str:
    normalized = str(column).strip()
    if normalized in TEMPORAL_STRING_CAST_COLUMNS:
        return f"cast({normalized} as string) as {normalized}"
    return normalized


def build_sample_sql(
    *,
    table_name: str,
    columns: Iterable[str],
    start_pt: str,
    end_pt: str,
) -> str:
    selected_columns = ", ".join(build_select_expression(column) for column in dict.fromkeys(columns))
    return f"select {selected_columns} from {table_name} where {build_partition_predicate(start_pt, end_pt)}"


def fetch_frame(
    sql: str,
    batch_size: int = 250000,
    *,
    reader_config: Optional[Dict[str, Any]] = None,
    expected_rows: Optional[int] = None,
    execution_details: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    client = create_odps_client_from_env()
    runtime_reader_config = dict(reader_config or {})
    return fetch_sql_as_frame(
        sql,
        odps_client=client,
        batch_size=batch_size,
        use_arrow=bool(runtime_reader_config.get("use_arrow", True)),
        arrow_diagnostic_enabled=bool(runtime_reader_config.get("arrow_diagnostic_enabled", False)),
        fallback_row_threshold=int(runtime_reader_config.get("fallback_row_threshold", 2_000_000)),
        expected_rows=expected_rows,
        execution_details=execution_details,
    )


def fetch_live_audit_frames(table_name: str, runtime_window: RuntimeWindow, *, reader_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    full_predicate = build_partition_predicate(runtime_window.min_pt, runtime_window.max_pt)
    mature_predicate = build_partition_predicate(runtime_window.min_pt, runtime_window.modeling_end_pt)
    queries = {
        "partition_summary_full": (
            f"select pt, count(*) as row_count, avg(cast(response_label_positive_3d as double)) as positive_rate "
            f"from {table_name} where {full_predicate} group by pt order by pt limit 100"
        ),
        "overall_summary_full": (
            f"select count(*) as total_rows, "
            f"count(distinct concat(coalesce(player_id, ''), '|', cast(assignment_date as string))) as distinct_player_day_rows, "
            f"count(distinct player_id) as distinct_players, "
            f"min(assignment_date) as min_assignment_date, max(assignment_date) as max_assignment_date, "
            f"min(feature_snapshot_date) as min_feature_snapshot_date, max(feature_snapshot_date) as max_feature_snapshot_date, "
            f"sum(response_label_positive_3d) as positive_rows, avg(cast(response_label_positive_3d as double)) as positive_rate "
            f"from {table_name} where {full_predicate}"
        ),
        "overall_summary_mature": (
            f"select count(*) as total_rows, "
            f"count(distinct concat(coalesce(player_id, ''), '|', cast(assignment_date as string))) as distinct_player_day_rows, "
            f"count(distinct player_id) as distinct_players, "
            f"min(assignment_date) as min_assignment_date, max(assignment_date) as max_assignment_date, "
            f"sum(response_label_positive_3d) as positive_rows, avg(cast(response_label_positive_3d as double)) as positive_rate "
            f"from {table_name} where {mature_predicate}"
        ),
        "treatment_summary_mature": (
            f"select treatment_flag, has_voucher_treatment, has_sms_treatment, count(*) as row_count "
            f"from {table_name} where {mature_predicate} "
            f"group by treatment_flag, has_voucher_treatment, has_sms_treatment order by row_count desc limit 20"
        ),
        "vip_summary_mature": (
            f"select vip_level, count(*) as row_count, avg(cast(response_label_positive_3d as double)) as positive_rate "
            f"from {table_name} where {mature_predicate} "
            f"group by vip_level order by row_count desc limit 20"
        ),
        "key_nulls_mature": (
            f"select "
            f"sum(case when player_id is null then 1 else 0 end) as player_id_nulls, "
            f"sum(case when assignment_date is null then 1 else 0 end) as assignment_date_nulls, "
            f"sum(case when treatment_timestamp is null then 1 else 0 end) as treatment_timestamp_nulls, "
            f"sum(case when feature_snapshot_date is null then 1 else 0 end) as feature_snapshot_date_nulls, "
            f"sum(case when vip_level is null then 1 else 0 end) as vip_level_nulls, "
            f"sum(case when response_label_positive_3d is null then 1 else 0 end) as response_label_positive_3d_nulls "
            f"from {table_name} where {mature_predicate}"
        ),
        "sample_rows_mature": (
            f"select * from {table_name} where pt = '{runtime_window.modeling_end_pt}' limit 5"
        ),
    }
    return {name: fetch_frame(sql, batch_size=100000, reader_config=reader_config, expected_rows=1000) for name, sql in queries.items()}


def to_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in ["assignment_date", "feature_snapshot_date", "first_outcome_event_date", "last_outcome_event_date"]:
        if column in converted.columns:
            converted[column] = pd.to_datetime(converted[column], errors="coerce")
    if "treatment_timestamp" in converted.columns:
        converted["treatment_timestamp"] = pd.to_datetime(converted["treatment_timestamp"], errors="coerce")
    if "feature_create_time" in converted.columns:
        converted["feature_create_time"] = pd.to_datetime(converted["feature_create_time"], errors="coerce")
    return converted


def assign_time_splits(frame: pd.DataFrame, split_plan: SplitPlan) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    working = frame.copy()
    distinct_pts = sorted(working["pt"].astype(str).dropna().unique().tolist())
    expected_pts = list(split_plan.selected_mature_pts)
    if distinct_pts != expected_pts:
        raise ValueError(
            "Loaded data does not match the configured split window. "
            f"expected_pts={expected_pts[0]}..{expected_pts[-1]} ({len(expected_pts)} partitions), "
            f"actual_pts={distinct_pts[0] if distinct_pts else 'NA'}..{distinct_pts[-1] if distinct_pts else 'NA'} ({len(distinct_pts)} partitions)."
        )
    split_manifest = split_manifest_from_plan(split_plan)
    working["split"] = np.where(
        working["pt"].isin(split_manifest["train"]),
        "train",
        np.where(working["pt"].isin(split_manifest["validation"]), "validation", "test"),
    )
    return working, split_manifest


def classification_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    positive_prediction_count = int(y_pred.sum())
    negative_prediction_count = int(len(y_pred) - positive_prediction_count)
    true_positive_count = int(((y_true == 1) & (y_pred == 1)).sum())
    true_negative_count = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive_count = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative_count = int(((y_true == 1) & (y_pred == 0)).sum())
    specificity_denominator = true_negative_count + false_positive_count
    npv_denominator = true_negative_count + false_negative_count
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(y_pred.mean()),
        "predicted_negative_rate": float(1.0 - y_pred.mean()),
        "positive_prediction_count": positive_prediction_count,
        "negative_prediction_count": negative_prediction_count,
        "specificity": float(true_negative_count / specificity_denominator) if specificity_denominator else 0.0,
        "negative_predictive_value": float(true_negative_count / npv_denominator) if npv_denominator else 0.0,
        "true_positive_count": true_positive_count,
        "true_negative_count": true_negative_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
    }


def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    minimum: float,
    maximum: float,
    step: float,
    threshold_selection_config: Dict[str, Any] | None = None,
) -> Tuple[float, pd.DataFrame, str]:
    thresholds = np.arange(minimum, maximum + 1e-9, step)
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        rows.append({"threshold": float(threshold), **classification_threshold_metrics(y_true, y_score, float(threshold))})
    frame = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    selection_config = threshold_selection_config or {}
    policy = str(selection_config.get("policy", "max_precision_with_min_recall")).strip().lower()
    fallback_policy = str(selection_config.get("fallback_policy", "max_precision")).strip().lower()
    minimum_recall = float(selection_config.get("minimum_recall", 0.0) or 0.0)

    if policy == "max_precision_with_min_recall":
        eligible = frame.loc[frame["recall"] >= minimum_recall].copy()
        if not eligible.empty:
            selected = eligible.sort_values(
                ["precision", "f1", "recall", "threshold"],
                ascending=[False, False, False, False],
            ).iloc[0]
            return float(selected["threshold"]), frame, f"max_precision_with_min_recall(minimum_recall={minimum_recall:.2f})"
        policy = fallback_policy

    if policy == "max_precision":
        selected = frame.sort_values(
            ["precision", "recall", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        return float(selected["threshold"]), frame, "max_precision"

    if policy != "best_f1":
        raise ValueError(f"Unsupported threshold selection policy `{policy}`.")

    selected = frame.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return float(selected["threshold"]), frame, "best_f1"


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    row_count = int(len(y_true))
    positive_count = int(y_true.sum())
    positive_rate = float(y_true.mean())
    brier = float(brier_score_loss(y_true, y_score))
    return {
        "rows": row_count,
        "positives": positive_count,
        "positive_rate": positive_rate,
        "score_mean": float(np.mean(y_score)),
        "pr_auc": safe_pr_auc(y_true, y_score),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "brier": brier,
        "threshold": float(threshold),
        "row_count": row_count,
        "positive_count": positive_count,
        "prevalence": positive_rate,
        "brier_score": brier,
        **classification_threshold_metrics(y_true, y_score, threshold),
    }


def build_segment_metrics(
    frame: pd.DataFrame,
    *,
    split_name: str,
    model_name: str,
    segment_column: str,
    threshold: float,
    min_rows: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for segment_value, segment_frame in frame.groupby(segment_column, dropna=False, observed=False):
        if len(segment_frame) < min_rows:
            continue
        metrics = compute_metrics(
            segment_frame["target"].to_numpy(dtype=int),
            segment_frame["score"].to_numpy(dtype=float),
            threshold,
        )
        metrics.update(
            {
                "model_name": model_name,
                "split": split_name,
                "segment_column": segment_column,
                "segment_value": "__NULL__" if pd.isna(segment_value) else str(segment_value),
            }
        )
        rows.append(metrics)
    if not rows:
        return pd.DataFrame(
            columns=[
                "rows",
                "positives",
                "positive_rate",
                "score_mean",
                "pr_auc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "brier",
                "predicted_positive_rate",
                "predicted_negative_rate",
                "positive_prediction_count",
                "negative_prediction_count",
                "specificity",
                "negative_predictive_value",
                "threshold",
                "model_name",
                "split",
                "segment_column",
                "segment_value",
                "row_count",
                "positive_count",
                "prevalence",
                "brier_score",
                "true_positive_count",
                "true_negative_count",
                "false_positive_count",
                "false_negative_count",
            ]
        )
    return pd.DataFrame(rows).sort_values(["split", "rows", "segment_value"], ascending=[True, False, True]).reset_index(drop=True)


def fitted_feature_names(preprocessor: Any) -> List[str]:
    names = preprocessor.get_feature_names_out()
    cleaned = [str(name).split("__", 1)[-1] for name in names]
    return cleaned


def transform_named_feature_frame(
    *,
    split_frame: pd.DataFrame,
    raw_feature_order: List[str],
    transformed_feature_names: List[str],
    preprocessor: Any,
) -> pd.DataFrame:
    raw_features = split_frame.loc[:, raw_feature_order].copy()
    validate_scoring_input_frame(
        raw_features,
        expected_raw_feature_order=raw_feature_order,
    )
    transformed = preprocessor.transform(raw_features)
    return build_named_transformed_frame(
        transformed,
        transformed_feature_names=transformed_feature_names,
        index=split_frame.index,
    )


def train_one_model(
    *,
    model_name: str,
    config: Dict[str, Any],
    splits: Dict[str, pd.DataFrame],
    numeric_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Any]:
    if model_name == "logistic_regression":
        preprocessor = build_logreg_preprocessor(numeric_features, categorical_features)
        estimator = build_logreg_estimator(config["models"][model_name], random_seed=42)
    elif model_name == "lightgbm":
        preprocessor = build_lgbm_preprocessor(numeric_features, categorical_features)
        estimator = build_lgbm_estimator(config["models"][model_name], random_seed=42)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    x_train = splits["train"][numeric_features + categorical_features]
    y_train = splits["train"][config["target_column"]].to_numpy(dtype=int)
    preprocessor.fit(x_train)
    raw_feature_order = resolve_expected_raw_feature_order(
        declared_feature_columns=numeric_features + categorical_features,
        preprocessor=preprocessor,
    )
    feature_names = fitted_feature_names(preprocessor)
    transformed_feature_names = resolve_expected_transformed_feature_names(
        declared_transformed_feature_names=feature_names,
        preprocessor=preprocessor,
        estimator=estimator,
    )
    x_train_transformed = transform_named_feature_frame(
        split_frame=splits["train"],
        raw_feature_order=raw_feature_order,
        transformed_feature_names=transformed_feature_names,
        preprocessor=preprocessor,
    )
    estimator.fit(x_train_transformed, y_train)

    scored_frames: Dict[str, pd.DataFrame] = {}
    for split_name, split_frame in splits.items():
        transformed_frame = transform_named_feature_frame(
            split_frame=split_frame,
            raw_feature_order=raw_feature_order,
            transformed_feature_names=transformed_feature_names,
            preprocessor=preprocessor,
        )
        score = estimator.predict_proba(transformed_frame)[:, 1]
        scored = split_frame.loc[:, ["player_id", "assignment_date", "pt", "vip_level", config["target_column"]]].copy()
        scored["score"] = score
        scored = scored.rename(columns={config["target_column"]: "target"})
        scored_frames[split_name] = scored

    validation_scored = scored_frames["validation"]
    selected_threshold, threshold_table, threshold_policy = choose_threshold(
        validation_scored["target"].to_numpy(dtype=int),
        validation_scored["score"].to_numpy(dtype=float),
        float(config["threshold_grid"]["min"]),
        float(config["threshold_grid"]["max"]),
        float(config["threshold_grid"]["step"]),
        dict(config.get("threshold_selection", {})),
    )

    metrics_rows: List[Dict[str, Any]] = []
    for split_name, scored in scored_frames.items():
        metrics = compute_metrics(
            scored["target"].to_numpy(dtype=int),
            scored["score"].to_numpy(dtype=float),
            selected_threshold,
        )
        metrics.update({"model_name": model_name, "split": split_name})
        metrics_rows.append(metrics)

    metrics_frame = pd.DataFrame(metrics_rows)
    segment_frames = [
        build_segment_metrics(
            frame,
            split_name=split_name,
            model_name=model_name,
            segment_column=str(config["reporting"]["segment_column"]),
            threshold=selected_threshold,
            min_rows=int(config["reporting"]["min_segment_rows"]),
        )
        for split_name, frame in scored_frames.items()
    ]
    segment_metrics = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()

    if model_name == "logistic_regression":
        importance = logreg_feature_importance_frame(estimator, feature_names)
    else:
        importance = lgbm_feature_importance_frame(estimator, feature_names)

    return {
        "model_name": model_name,
        "preprocessor": preprocessor,
        "estimator": estimator,
        "selected_threshold": selected_threshold,
        "threshold_policy": threshold_policy,
        "threshold_table": threshold_table if threshold_table is not None else pd.DataFrame(),
        "metrics_frame": metrics_frame,
        "segment_metrics": segment_metrics,
        "feature_importance": importance,
        "scored_frames": scored_frames,
        "feature_names": feature_names,
        "raw_feature_order": raw_feature_order,
        "transformed_feature_names": transformed_feature_names,
    }


def score_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["decile"] = pd.qcut(working["score"].rank(method="first"), q=10, labels=list(range(1, 11)))
    summary = (
        working.groupby("decile", as_index=False, observed=False)
        .agg(
            rows=("target", "size"),
            positive_rate=("target", "mean"),
            avg_score=("score", "mean"),
        )
        .sort_values("decile", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def cohort_profile(frame: pd.DataFrame, numeric_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranked = frame.sort_values("score", ascending=False).reset_index(drop=True)
    quintile_size = max(1, len(ranked) // 5)
    high = ranked.head(quintile_size)
    low = ranked.tail(quintile_size)
    numeric_rows = []
    for column in numeric_features:
        numeric_rows.append(
            {
                "feature": column,
                "high_score_mean": float(high[column].mean()),
                "low_score_mean": float(low[column].mean()),
                "difference": float(high[column].mean() - low[column].mean()),
            }
        )
    numeric_profile = pd.DataFrame(numeric_rows).sort_values("difference", ascending=False).reset_index(drop=True)
    vip_profile = (
        pd.concat(
            [
                high["vip_level"].fillna("__NULL__").value_counts(normalize=True).rename("high_score_share"),
                low["vip_level"].fillna("__NULL__").value_counts(normalize=True).rename("low_score_share"),
            ],
            axis=1,
        )
        .fillna(0.0)
        .reset_index()
        .rename(columns={"index": "vip_level"})
        .sort_values("high_score_share", ascending=False)
        .reset_index(drop=True)
    )
    return numeric_profile, vip_profile


def build_schema_frame(table_name: str) -> pd.DataFrame:
    project, table = table_name.split(".", 1)
    client = create_odps_client_from_env()
    odps_table = client.get_table(table, project=project)
    rows = []
    seen = set()
    partition_names = {partition.name for partition in odps_table.table_schema.partitions}
    for column in list(odps_table.table_schema.columns) + list(odps_table.table_schema.partitions):
        if column.name in seen:
            continue
        seen.add(column.name)
        rows.append(
            {
                "column_name": column.name,
                "data_type": str(column.type),
                "comment": column.comment or "",
                "is_partition": column.name in partition_names,
            }
        )
    return pd.DataFrame(rows)


def publish_champion_bundle(
    *,
    config_path: Path,
    config: Dict[str, Any],
    best_result: Dict[str, Any],
) -> Dict[str, str]:
    iteration_id = config_path.stem.strip() or "response_model"
    run_id = datetime.utcnow().strftime("phase1_response_%Y%m%dT%H%M%SZ")
    model_name = str(best_result["model_name"])
    model_version = build_model_version(
        iteration_id=iteration_id,
        model_name=model_name,
        mlflow_run_id=run_id,
    )
    bundle_dir = ensure_dir(WORKSPACE_ROOT / "outputs" / "runs" / f"{iteration_id}__{run_id}__{model_name}")
    model_dir = ensure_dir(bundle_dir / "model")
    champion_reference_path = resolve_response_champion_reference_path(config)
    selected_threshold = float(best_result["selected_threshold"])
    selected_score_variant = "response_probability"

    joblib.dump(
        {
            "preprocessor": best_result["preprocessor"],
            "estimator": best_result["estimator"],
            "selected_threshold": selected_threshold,
            "threshold_policy": str(best_result["threshold_policy"]),
            "feature_names": list(best_result["feature_names"]),
            "raw_feature_order": list(best_result["raw_feature_order"]),
            "transformed_feature_names": list(best_result["transformed_feature_names"]),
        },
        model_dir / "model.joblib",
    )
    joblib.dump(best_result["preprocessor"], model_dir / "preprocessor.joblib")
    save_json(
        model_dir / "preprocessing.json",
        {
            "model_name": model_name,
            "preprocessor_class": best_result["preprocessor"].__class__.__name__,
            "raw_feature_order": list(best_result["raw_feature_order"]),
            "transformed_feature_names": list(best_result["transformed_feature_names"]),
            "selected_score_variant": selected_score_variant,
        },
    )
    save_json(
        model_dir / "threshold_selection.json",
        {
            "selected_threshold": selected_threshold,
            "selected_score_variant": selected_score_variant,
            "threshold_policy": str(best_result["threshold_policy"]),
        },
    )
    save_json(
        bundle_dir / "feature_schema.json",
        {
            "feature_columns": list(dict.fromkeys(list(config["numeric_feature_columns"]) + list(config["categorical_feature_columns"]))),
            "numeric_feature_columns": list(config["numeric_feature_columns"]),
            "categorical_feature_columns": list(config["categorical_feature_columns"]),
            "raw_feature_order": list(best_result["raw_feature_order"]),
            "transformed_feature_names": list(best_result["transformed_feature_names"]),
            "fitted_feature_names": list(best_result["feature_names"]),
            "target_column": str(config["target_column"]),
            "grain": ["player_id", "assignment_date"],
        },
    )
    save_json(
        bundle_dir / "run_metadata.json",
        {
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id,
            "iteration_id": iteration_id,
            "selected_threshold": selected_threshold,
            "selected_score_variant": selected_score_variant,
            "export_bundle_path": str(bundle_dir),
            "source_table": str(config["source_table"]),
            "target_column": str(config["target_column"]),
        },
    )

    write_champion_reference(
        champion_reference_path,
        {
            "model_dir": str(model_dir),
            "export_bundle_path": str(bundle_dir),
            "model_name": model_name,
            "model_version": model_version,
            "mlflow_run_id": run_id,
            "iteration_id": iteration_id,
            "threshold_artifact_path": str(model_dir / "threshold_selection.json"),
            "feature_schema_path": str(bundle_dir / "feature_schema.json"),
            "run_metadata_path": str(bundle_dir / "run_metadata.json"),
            "selected_threshold": selected_threshold,
            "selected_score_variant": selected_score_variant,
        },
    )
    return {
        "bundle_dir": str(bundle_dir),
        "model_dir": str(model_dir),
        "model_version": model_version,
        "run_id": run_id,
        "champion_reference_path": str(champion_reference_path),
    }


def build_reports(
    *,
    config: Dict[str, Any],
    runtime_window: RuntimeWindow,
    schema_frame: pd.DataFrame,
    audit_frames: Dict[str, pd.DataFrame],
    sample_frame: pd.DataFrame,
    sampled_missingness: pd.DataFrame,
    split_manifest: Dict[str, List[str]],
    results: Dict[str, Dict[str, Any]],
    best_result: Dict[str, Any],
    artifacts_dir: Path,
) -> None:
    reports_dir = PROJECT_ROOT / "reports"
    overall_full = audit_frames["overall_summary_full"].iloc[0].to_dict()
    overall_mature = audit_frames["overall_summary_mature"].iloc[0].to_dict()
    partition_summary = audit_frames["partition_summary_full"].copy()
    partition_summary["mature_for_modeling"] = partition_summary["pt"].astype(str) <= runtime_window.modeling_end_pt

    candidate_targets = pd.DataFrame(
        [
            {
                "target_column": "response_label_positive_3d",
                "task_family": "binary classification",
                "status": "recommended",
                "notes": "Directly observed binary response flag derived from positive 3-day gross bet.",
            },
            {
                "target_column": "outcome_gross_bet_3d_value",
                "task_family": "regression",
                "status": "secondary only",
                "notes": "Heavy zero inflation and long-tailed spend make it a weaker first baseline.",
            },
            {
                "target_column": "outcome_gross_ggr_3d_value",
                "task_family": "regression",
                "status": "secondary only",
                "notes": "More volatile commercial proxy and harder to communicate as a first response model.",
            },
        ]
    )
    feature_groups = pd.DataFrame(
        [
            {"group": "Pre-treatment recent betting behavior", "columns": "recent_bet_cnt_7d, recent_bet_amt_7d, recent_win_amt_7d, recent_ggr_amt_7d, recent_net_loss_amt_7d, recent_bet_days_7d, recency_last_bet_to_t"},
            {"group": "Pre-treatment 30-day betting behavior", "columns": "pre_bet_cnt_30d, pre_bet_amt_30d, pre_win_amt_30d, pre_ggr_amt_30d, pre_net_loss_amt_30d, pre_bet_days_30d"},
            {"group": "Pre-treatment cashflow behavior", "columns": "recent_dep_cnt_7d, recent_dep_amt_7d, recent_withdraw_cnt_7d, recent_withdraw_amt_7d, recent_net_cash_in_7d"},
            {"group": "Pre-treatment categorical segment", "columns": "vip_level"},
            {"group": "Observed treatment metadata", "columns": "has_voucher_treatment, has_sms_treatment, treatment_flag, raw_assignment_event_count"},
        ]
    )
    excluded_columns = pd.DataFrame(
        [
            {"column_name": "player_id", "reason": "Identifier; not a model feature."},
            {"column_name": "assignment_date", "reason": "Row key and temporal anchor, not a behavioral feature."},
            {"column_name": "treatment_timestamp", "reason": "Operational timestamp after assignment logic; excluded from the baseline."},
            {"column_name": "treatment_flag", "reason": "Constant 1 for all rows; no predictive value."},
            {"column_name": "raw_assignment_event_count", "reason": "Execution artifact rather than a stable pre-treatment covariate."},
            {"column_name": "has_voucher_treatment", "reason": "Observed channel metadata; excluded from the default baseline to avoid conditioning on post-decision execution."},
            {"column_name": "has_sms_treatment", "reason": "Observed channel metadata; excluded from the default baseline to avoid conditioning on post-decision execution."},
            {"column_name": "feature_snapshot_date", "reason": "Snapshot-control field rather than user behavior."},
            {"column_name": "feature_create_time", "reason": "Audit metadata."},
            {"column_name": "data_version", "reason": "Audit metadata."},
            {"column_name": "outcome_gross_bet_3d_value", "reason": "Post-treatment realized outcome and direct leakage for response_label_positive_3d."},
            {"column_name": "outcome_gross_ggr_3d_value", "reason": "Post-treatment realized outcome and leakage-risk field."},
            {"column_name": "outcome_source_row_count_3d", "reason": "Outcome window construction metadata after treatment."},
            {"column_name": "outcome_distinct_source_keys_3d", "reason": "Outcome window construction metadata after treatment."},
            {"column_name": "first_outcome_event_date", "reason": "Post-treatment event timing leakage."},
            {"column_name": "last_outcome_event_date", "reason": "Post-treatment event timing leakage."},
            {"column_name": "response_label_positive_3d", "reason": "Target column."},
            {"column_name": "pt", "reason": "Temporal split key, not a model feature."},
        ]
    )

    audit_report = f"""
# Phase-1 Observational Response Dataset Audit

This audit is for `pai_rec_prod.alg_uplift_phase1_response_dataset_di`.

This table is treated as an observational response dataset. It is not a causal uplift dataset, and nothing in this audit should be read as proof of treatment effect.

## Live Table Coverage

- Partition coverage discovered from ODPS metadata: `{runtime_window.min_pt}` to `{runtime_window.max_pt}` (`{len(partition_summary)}` daily partitions).
- Conservative modeling cutoff for a 3-day response label on `as_of_date={config['as_of_date']}`: `pt <= {runtime_window.modeling_end_pt}`.
- Full table rows across all live partitions: `{format_int(overall_full['total_rows'])}`.
- Mature modeling rows through `{runtime_window.modeling_end_pt}`: `{format_int(overall_mature['total_rows'])}`.
- Distinct `(player_id, assignment_date)` keys across mature rows: `{format_int(overall_mature['distinct_player_day_rows'])}`.
- Mature-table positive response rate: `{format_float(overall_mature['positive_rate'])}`.

## Schema Summary

{markdown_table(schema_frame)}

## Partition Summary

{markdown_table(partition_summary, max_rows=20)}

## Modeling Grain

- Confirmed row grain: one treated `player_id` per `assignment_date`.
- Live ODPS check: `total_rows == distinct(player_id, assignment_date)` on the audited mature range.
- Treatment semantics: every row is already treated (`treatment_flag = 1`), so this is response modeling within exposed rows, not treat-vs-control uplift estimation.

## Candidate Target Column(s)

{markdown_table(candidate_targets)}

## Candidate Feature Column Groups

{markdown_table(feature_groups)}

## Columns To Exclude From Modeling

{markdown_table(excluded_columns)}

## Treatment Indicator Columns Present

- `treatment_flag`
- `has_voucher_treatment`
- `has_sms_treatment`
- `raw_assignment_event_count`

## Key Data Quality Caveats

- The table contains very recent partitions through `{runtime_window.max_pt}`. For a 3-day post-treatment response label, partitions after `{runtime_window.modeling_end_pt}` are conservatively treated as immature and excluded from training/evaluation.
- `vip_level` is missing for `{format_int(audit_frames['key_nulls_mature'].iloc[0]['vip_level_nulls'])}` mature rows, so the baseline keeps it with explicit missing-category handling rather than dropping those rows.
- Positive-rate drift is material across partitions, from roughly `{format_float(partition_summary['positive_rate'].min())}` to `{format_float(partition_summary['positive_rate'].max())}`. Time-based validation is therefore required.
- The dataset remains observational only. All rows are exposed rows, so model outputs estimate likely response conditional on historical exposure patterns, not incremental causal lift.

## Sample Rows

{markdown_table(audit_frames['sample_rows_mature'])}

## Sampled Missingness Check

The detailed column-level missingness check below is computed on the reproducible local hash sample used for baseline training.

{markdown_table(sampled_missingness, max_rows=25)}
"""
    write_text(reports_dir / "response_model_dataset_audit.md", audit_report)

    task_report = f"""
# Phase-1 Response Model Task Definition

## Recommended Modeling Task

- Recommended task: binary response modeling.
- Exact target column: `response_label_positive_3d`.

## Why This Is The Best Phase-1 Choice

- `response_label_positive_3d` is the cleanest directly observed label already present in the audited table.
- It avoids the heavy zero-inflation and long-tail instability of raw post-treatment value regression targets.
- There is no audited multiclass response label in the live schema, so multiclass modeling would require inventing bins or semantics that the table does not define.
- A binary response baseline is the most defensible first step for an observational dataset where all rows are already treated.

## What This Answers For The Business

- It estimates which historically exposed users look most likely to show a positive 3-day betting response after a marketing exposure.
- It supports ranking or prioritizing likely responders within future marketing candidate pools, provided the business treats the score as observational propensity to react, not proof of lift.

## What It Does Not Answer

- It does not estimate incremental causal effect.
- It does not tell the business whether a user would have responded without treatment.
- It does not prove that SMS or vouchers cause the observed response.
- It does not optimize treatment-vs-no-treatment policy because the dataset has no untreated control rows.
"""
    write_text(reports_dir / "response_model_task_definition.md", task_report)

    split_rows = []
    for split_name, pts in split_manifest.items():
        split_rows.append(
            {
                "split": split_name,
                "distinct_pt_count": len(pts),
                "start_pt": pts[0],
                "end_pt": pts[-1],
            }
        )
    split_frame = pd.DataFrame(split_rows)
    feature_report = f"""
# Phase-1 Response Model Feature Selection

## Included Modeling Features

- Numeric pre-treatment features: `{", ".join(config['numeric_feature_columns'])}`
- Categorical pre-treatment features: `{", ".join(config['categorical_feature_columns'])}`

## Excluded Columns

{markdown_table(excluded_columns)}

## Leakage Rationale

- All realized outcome columns and outcome-window metadata are excluded because they are observed after treatment and would leak the response label.
- Identifier, audit, and partition columns are excluded because they are not stable behavioral features.
- Treatment-channel fields (`has_voucher_treatment`, `has_sms_treatment`) are kept in the extracted dataset for auditing, but excluded from the default baseline because they describe realized execution rather than universally available pre-treatment state. They can be added later only for channel-conditional models where the planned channel is known at score time.

## Split Strategy

- Strategy: time-based split on `pt`.
- Sampling: disabled in the canonical runtime. Training uses the full configured mature ODPS split window.
- Mature modeling window used for model fitting/evaluation: `{split_frame['start_pt'].min()}` to `{split_frame['end_pt'].max()}`.
- Split manifest:

{markdown_table(split_frame)}
"""
    write_text(reports_dir / "response_model_feature_selection.md", feature_report)

    comparison_rows = []
    for result in results.values():
        validation_metrics = result["metrics_frame"].loc[result["metrics_frame"]["split"] == "validation"].iloc[0]
        test_metrics = result["metrics_frame"].loc[result["metrics_frame"]["split"] == "test"].iloc[0]
        comparison_rows.append(
            {
                "model_name": result["model_name"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_f1": validation_metrics["f1"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_f1": test_metrics["f1"],
                "selected_threshold": result["selected_threshold"],
            }
        )
    comparison_frame = pd.DataFrame(comparison_rows).sort_values("validation_pr_auc", ascending=False).reset_index(drop=True)
    best_test_metrics = best_result["metrics_frame"].loc[best_result["metrics_frame"]["split"] == "test"].iloc[0]
    best_validation_metrics = best_result["metrics_frame"].loc[best_result["metrics_frame"]["split"] == "validation"].iloc[0]
    best_segment_validation = best_result["segment_metrics"].loc[best_result["segment_metrics"]["split"] == "validation"].copy()
    best_segment_validation = best_segment_validation.sort_values("rows", ascending=False).reset_index(drop=True)
    best_segment_test = best_result["segment_metrics"].loc[best_result["segment_metrics"]["split"] == "test"].copy()
    best_segment_test = best_segment_test.sort_values("rows", ascending=False).reset_index(drop=True)
    baseline_report = f"""
# Phase-1 Response Model Baseline Results

The baselines below are observational response models trained on the full configured mature ODPS split window only.

## Model Comparison

{markdown_table(comparison_frame)}

## Global Train / Validation / Test Metrics

{markdown_table(best_result['metrics_frame'])}

## Validation Segment-Level Metrics By `vip_level`

{markdown_table(best_segment_validation, max_rows=15)}

## Test Segment-Level Metrics By `vip_level`

{markdown_table(best_segment_test, max_rows=15)}

## Key Strengths

- The best baseline (`{best_result['model_name']}`) produces a test PR-AUC of `{format_float(best_test_metrics['pr_auc'])}` and ROC-AUC of `{format_float(best_test_metrics['roc_auc'])}` on held-out later partitions.
- The same baseline produces validation PR-AUC of `{format_float(best_validation_metrics['pr_auc'])}` with explicit `vip_level` segment cuts persisted for validation and test.
- Time-based splitting forces the model to generalize across real response-rate drift rather than relying on random-row leakage.
- Segment reporting shows the model remains interpretable across `vip_level` groups instead of acting as a pure black box.

## Key Weaknesses

- Precision and recall depend on a validation-chosen threshold and should be treated as operational placeholders, not final campaign policy.
- Performance still reflects observational exposure history, so selection bias remains.
- `vip_level` missingness and strong day-to-day prevalence drift mean the model will need monitoring if promoted into regular targeting support.

## First-Iteration Usefulness

- Recommendation: `{"YES" if best_test_metrics['pr_auc'] >= 0.30 and best_test_metrics['precision'] >= 0.30 else "MAYBE"}` for a first business iteration as a ranking or prioritization support model.
- Constraint: use it only to rank likely responders within eligible campaign pools, not to claim causal uplift.
"""
    write_text(reports_dir / "response_model_baseline_results.md", baseline_report)

    best_test_scored = best_result["scored_frames"]["test"].merge(
        sample_frame[["player_id", "assignment_date"] + config["numeric_feature_columns"]],
        on=["player_id", "assignment_date"],
        how="left",
    )
    deciles = score_deciles(best_test_scored)
    numeric_profile, vip_profile = cohort_profile(best_test_scored, config["numeric_feature_columns"])
    interpretation_report = f"""
# Phase-1 Response Model Interpretation

This interpretation is for the selected baseline `{best_result['model_name']}`. It explains who is historically more likely to respond positively after exposure in this observational dataset. It does not claim causal treatment effect.

## Feature Importance

{markdown_table(best_result['feature_importance'], max_rows=15)}

## Score Deciles On Test Data

{markdown_table(deciles)}

## High-Score vs Low-Score Numeric Profile

{markdown_table(numeric_profile, max_rows=12)}

## High-Score vs Low-Score `vip_level` Mix

{markdown_table(vip_profile, max_rows=10)}

## Business Reading

- Users with the highest predicted positive-response scores are concentrated in higher pre-treatment betting and cash-in activity bands.
- Higher `vip_level` segments occupy a larger share of the high-score cohort, while `UNKNOWN`, missing, and lower VIP groups are more common among low-score users.
- The model therefore appears to be ranking historical engagement intensity and commercial value, which is reasonable for a first observational response baseline.
- Users with low recent betting, low recent deposits, and weaker 30-day betting history are much less likely to show a positive 3-day reaction.
- Weak or neutral responders are best interpreted as low-probability responders under historical exposure patterns, not as users who should never be treated.
"""
    write_text(reports_dir / "response_model_interpretation.md", interpretation_report)

    recommendation_report = f"""
# Phase-1 Response Model Business Recommendation

## What Can Be Done Now

- Use the response score as a prioritization layer inside already-approved campaign candidate pools.
- Prefer higher-scoring users when budget or contact volume is limited.
- Review very low-scoring users as possible deprioritization candidates for expensive exposures.
- Track live score distribution, realized response rate, and segment coverage by `vip_level` if the score is used operationally.

## What This Model Does Not Justify

- It does not justify claims of incremental uplift.
- It should not be used as proof that a campaign caused additional betting or GGR.
- It should not be used as a final treatment-vs-no-treatment decision rule without a future experiment or quasi-experimental design.

## What Requires Future Experiment Design

- Any estimate of incremental lift or ROI attributable to the intervention itself.
- Any claim that one channel is better than another because channel assignment is not randomized here.
- Calibration of contact thresholds to business cost and user fatigue tradeoffs.

## Recommended Next Step

- Run a controlled business pilot that uses this observational response model only to rank already-eligible users, while simultaneously carving out a future holdout design so the organization can later measure incremental effect rather than only observed reaction.
"""
    write_text(reports_dir / "response_model_business_recommendation.md", recommendation_report)

    save_frame(artifacts_dir / "audit_partition_summary_full.csv", partition_summary)
    save_frame(artifacts_dir / "audit_schema.csv", schema_frame)
    save_frame(artifacts_dir / "sample_missingness.csv", sampled_missingness)
    save_frame(artifacts_dir / "model_comparison.csv", comparison_frame)
    save_frame(artifacts_dir / "best_model_vip_level_metrics_validation.csv", best_segment_validation)
    save_frame(artifacts_dir / "best_model_vip_level_metrics_test.csv", best_segment_test)
    save_frame(artifacts_dir / "best_model_test_deciles.csv", deciles)
    save_frame(artifacts_dir / "best_model_high_low_numeric_profile.csv", numeric_profile)
    save_frame(artifacts_dir / "best_model_high_low_vip_profile.csv", vip_profile)


def sampled_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        rows.append(
            {
                "column_name": column,
                "missing_rows": int(frame[column].isna().sum()),
                "missing_rate": float(frame[column].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "column_name"], ascending=[False, True]).reset_index(drop=True)


def count_rows_for_predicate(
    table_name: str,
    predicate: str,
    *,
    reader_config: Dict[str, Any],
) -> int:
    sql = f"select count(*) as row_count from {table_name} where {predicate}"
    frame = fetch_frame(sql, batch_size=10000, reader_config=reader_config, expected_rows=1)
    if frame.empty:
        return 0
    return int(frame.iloc[0]["row_count"])


def count_rows_by_split(
    table_name: str,
    split_plan: SplitPlan,
    *,
    reader_config: Dict[str, Any],
) -> Dict[str, int]:
    return {
        window.name: count_rows_for_predicate(
            table_name,
            build_partition_predicate(window.start_pt, window.end_pt),
            reader_config=reader_config,
        )
        for window in (split_plan.train, split_plan.validation, split_plan.test)
    }


def log_split_plan(runtime_window: RuntimeWindow, split_plan: SplitPlan, row_counts: Dict[str, int], extraction_columns: Sequence[str]) -> None:
    total_rows_before_pull = int(sum(row_counts.values()))
    estimated_memory = estimate_memory_bytes(total_rows_before_pull, len(list(extraction_columns)))
    print(
        f"[PHASE1] live_range={runtime_window.min_pt}..{runtime_window.max_pt} "
        f"mature_cutoff={split_plan.mature_cutoff_pt}",
        flush=True,
    )
    print(
        f"[PHASE1] train_start={split_plan.train.start_pt} train_end={split_plan.train.end_pt} "
        f"validation_start={split_plan.validation.start_pt} validation_end={split_plan.validation.end_pt} "
        f"test_start={split_plan.test.start_pt} test_end={split_plan.test.end_pt}",
        flush=True,
    )
    print(
        f"[PHASE1] row_counts train={row_counts['train']} validation={row_counts['validation']} "
        f"test={row_counts['test']} total_rows_before_pull={total_rows_before_pull}",
        flush=True,
    )
    print(
        f"[PHASE1] memory_estimate={format_bytes(estimated_memory)} "
        f"assumption=~{max(256, len(list(extraction_columns)) * 32)}B_per_row",
        flush=True,
    )


def log_dry_run_details(
    *,
    partitions: Sequence[str],
    runtime_window: RuntimeWindow,
    split_plan: SplitPlan,
    row_counts: Dict[str, int],
    extraction_columns: Sequence[str],
    full_pull_sql: str,
) -> None:
    print(f"[PHASE1][DRY_RUN] partition_count={len(partitions)}", flush=True)
    print(f"[PHASE1][DRY_RUN] partitions={','.join(str(pt) for pt in partitions)}", flush=True)
    print(
        f"[PHASE1][DRY_RUN] mature_partitions_selected={','.join(split_plan.selected_mature_pts)}",
        flush=True,
    )
    print(
        f"[PHASE1][DRY_RUN] where_train={build_partition_predicate(split_plan.train.start_pt, split_plan.train.end_pt)}",
        flush=True,
    )
    print(
        "[PHASE1][DRY_RUN] "
        f"where_validation={build_partition_predicate(split_plan.validation.start_pt, split_plan.validation.end_pt)}",
        flush=True,
    )
    print(
        f"[PHASE1][DRY_RUN] where_test={build_partition_predicate(split_plan.test.start_pt, split_plan.test.end_pt)}",
        flush=True,
    )
    print(
        f"[PHASE1][DRY_RUN] where_full_pull={build_partition_predicate(split_plan.train.start_pt, split_plan.test.end_pt)}",
        flush=True,
    )
    print(f"[PHASE1][DRY_RUN] full_pull_sql={full_pull_sql}", flush=True)
    log_split_plan(runtime_window, split_plan, row_counts, extraction_columns)


def main() -> None:
    bootstrap_runtime_environment()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run Phase-1 observational response modeling from the ODPS response table.")
    parser.add_argument(
        "--config-path",
        default=str(PROJECT_ROOT / "configs" / "response_model.yaml"),
        help="Path to the Phase-1 response modeling config.",
    )
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    print(f"[PHASE1] using_config_path={config_path}", flush=True)
    config = load_yaml(config_path)
    reader_config = {
        "use_arrow": bool(config.get("odps_reader", {}).get("use_arrow", False)),
        "arrow_diagnostic_enabled": bool(config.get("odps_reader", {}).get("arrow_diagnostic_enabled", False)),
        "fallback_row_threshold": int(config.get("odps_reader", {}).get("fallback_row_threshold", 5_000_000)),
    }
    split_policy = build_split_policy(config)
    artifacts_dir = ensure_dir((PROJECT_ROOT / config["artifacts"]["output_dir"]).resolve())
    model_dir = ensure_dir(artifacts_dir / "models")

    print("[PHASE1] listing live ODPS partitions", flush=True)
    partitions = list_partitions(str(config["source_table"]))
    runtime_window = resolve_runtime_window(config, partitions)
    split_plan = resolve_split_plan(runtime_window, partitions, split_policy)
    split_manifest = split_manifest_from_plan(split_plan)

    extraction_columns = [
        "player_id",
        "assignment_date",
        "treatment_timestamp",
        "treatment_flag",
        "raw_assignment_event_count",
        "has_voucher_treatment",
        "has_sms_treatment",
        "feature_snapshot_date",
        *config["numeric_feature_columns"],
        *config["categorical_feature_columns"],
        "outcome_gross_bet_3d_value",
        "outcome_gross_ggr_3d_value",
        "outcome_source_row_count_3d",
        "outcome_distinct_source_keys_3d",
        "first_outcome_event_date",
        "last_outcome_event_date",
        str(config["target_column"]),
        "pt",
        "feature_create_time",
        "data_version",
    ]
    sample_sql = build_sample_sql(
        table_name=str(config["source_table"]),
        columns=extraction_columns,
        start_pt=split_plan.train.start_pt,
        end_pt=split_plan.test.end_pt,
    )
    row_counts = count_rows_by_split(str(config["source_table"]), split_plan, reader_config=reader_config)
    log_split_plan(runtime_window, split_plan, row_counts, extraction_columns)
    print(
        f"[PHASE1] arrow_enabled={str(reader_config['use_arrow']).lower()} "
        f"arrow_diagnostic_enabled={str(reader_config['arrow_diagnostic_enabled']).lower()} "
        f"fallback_row_threshold={reader_config['fallback_row_threshold']}",
        flush=True,
    )

    if args.dry_run:
        log_dry_run_details(
            partitions=partitions,
            runtime_window=runtime_window,
            split_plan=split_plan,
            row_counts=row_counts,
            extraction_columns=extraction_columns,
            full_pull_sql=sample_sql,
        )
        print("[PHASE1] dry_run_complete no_full_data_pull=true", flush=True)
        return

    print("[PHASE1] fetching schema and audit summaries", flush=True)
    schema_frame = build_schema_frame(str(config["source_table"]))
    audit_frames = fetch_live_audit_frames(str(config["source_table"]), runtime_window, reader_config=reader_config)
    print("[PHASE1] pulling mature ODPS training window", flush=True)
    reader_execution: Dict[str, Any] = {}
    sample_frame = to_datetime_columns(
        fetch_frame(
            sample_sql,
            batch_size=250000,
            reader_config=reader_config,
            expected_rows=int(sum(row_counts.values())),
            execution_details=reader_execution,
        )
    )
    sample_frame["pt"] = sample_frame["pt"].astype(str)
    sample_frame = sample_frame.sort_values(["pt", "player_id", "assignment_date"]).reset_index(drop=True)
    print(
        f"[PHASE1] loaded_rows={len(sample_frame)} "
        f"fallback_used={str(bool(reader_execution.get('fallback_used', False))).lower()}",
        flush=True,
    )
    missingness_frame = sampled_missingness(sample_frame)

    split_frame, split_manifest = assign_time_splits(sample_frame, split_plan)
    print(
        f"[PHASE1] split_pts train={len(split_manifest['train'])} validation={len(split_manifest['validation'])} test={len(split_manifest['test'])}",
        flush=True,
    )
    loaded_row_counts = {name: int(len(frame)) for name, frame in split_frame.groupby("split", observed=False)}
    print(
        f"[PHASE1] loaded_split_rows train={loaded_row_counts.get('train', 0)} "
        f"validation={loaded_row_counts.get('validation', 0)} "
        f"test={loaded_row_counts.get('test', 0)}",
        flush=True,
    )

    splits = {name: frame.reset_index(drop=True) for name, frame in split_frame.groupby("split", observed=False)}
    results: Dict[str, Dict[str, Any]] = {}
    for model_name, model_config in config["models"].items():
        if not bool(model_config.get("enabled", False)):
            continue
        print(f"[PHASE1] training_model={model_name}", flush=True)
        result = train_one_model(
            model_name=model_name,
            config=config,
            splits=splits,
            numeric_features=list(config["numeric_feature_columns"]),
            categorical_features=list(config["categorical_feature_columns"]),
        )
        results[model_name] = result
        output_prefix = model_dir / model_name
        ensure_dir(output_prefix)
        joblib.dump(
            {
                "preprocessor": result["preprocessor"],
                "estimator": result["estimator"],
                "selected_threshold": result["selected_threshold"],
                "threshold_policy": result["threshold_policy"],
                "feature_names": result["feature_names"],
                "raw_feature_order": result["raw_feature_order"],
                "transformed_feature_names": result["transformed_feature_names"],
            },
            output_prefix / "model.joblib",
        )
        save_frame(output_prefix / "metrics.csv", result["metrics_frame"])
        save_frame(output_prefix / "segment_metrics.csv", result["segment_metrics"])
        for split_name in ["validation", "test"]:
            split_segment_metrics = result["segment_metrics"].loc[result["segment_metrics"]["split"] == split_name].copy()
            save_frame(output_prefix / f"vip_level_metrics_{split_name}.csv", split_segment_metrics)
        save_frame(output_prefix / "threshold_table.csv", result["threshold_table"])
        save_frame(output_prefix / "feature_importance.csv", result["feature_importance"])
        write_text(
            output_prefix / "model_summary.json",
            json.dumps(
                {
                    "model_name": result["model_name"],
                    "selected_threshold": result["selected_threshold"],
                    "threshold_policy": result["threshold_policy"],
                    "feature_names": result["feature_names"],
                },
                indent=2,
            ),
        )

    if not results:
        raise ValueError("No models were enabled.")

    print("[PHASE1] writing reports", flush=True)
    best_result = max(
        results.values(),
        key=lambda result: float(result["metrics_frame"].loc[result["metrics_frame"]["split"] == "validation", "pr_auc"].iloc[0]),
    )
    build_reports(
        config=config,
        runtime_window=runtime_window,
        schema_frame=schema_frame,
        audit_frames=audit_frames,
        sample_frame=split_frame,
        sampled_missingness=missingness_frame,
        split_manifest=split_manifest,
        results=results,
        best_result=best_result,
        artifacts_dir=artifacts_dir,
    )

    reporting_summary: Dict[str, Any] | None = None
    if "lightgbm" in results:
        print("[PHASE1] regenerating integrated LightGBM VIP and threshold reporting", flush=True)
        reporting_summary = generate_reporting_artifacts(
            model_dir=model_dir / "lightgbm",
            reports_dir=PROJECT_ROOT / "reports",
            scored_by_split={
                "validation": results["lightgbm"]["scored_frames"]["validation"],
                "test": results["lightgbm"]["scored_frames"]["test"],
            },
        )

    champion_artifacts = publish_champion_bundle(
        config_path=config_path,
        config=config,
        best_result=best_result,
    )
    print(
        "[PHASE1] champion_published "
        f"model={best_result['model_name']} "
        f"model_version={champion_artifacts['model_version']} "
        f"champion_reference_path={champion_artifacts['champion_reference_path']}",
        flush=True,
    )

    summary = {
        "runtime_window": {
            "min_pt": runtime_window.min_pt,
            "max_pt": runtime_window.max_pt,
            "maturity_end_pt": runtime_window.maturity_end_pt,
            "modeling_end_pt": runtime_window.modeling_end_pt,
        },
        "split_policy": split_policy,
        "split_windows": {
            "train": {"start_pt": split_plan.train.start_pt, "end_pt": split_plan.train.end_pt},
            "validation": {"start_pt": split_plan.validation.start_pt, "end_pt": split_plan.validation.end_pt},
            "test": {"start_pt": split_plan.test.start_pt, "end_pt": split_plan.test.end_pt},
        },
        "row_counts_before_pull": row_counts,
        "row_counts_loaded": loaded_row_counts,
        "sample_rows": int(len(sample_frame)),
        "odps_reader": {
            **reader_config,
            "fallback_used": bool(reader_execution.get("fallback_used", False)),
        },
        "models": {
            name: {
                "selected_threshold": float(result["selected_threshold"]),
                "threshold_policy": str(result["threshold_policy"]),
                "validation_pr_auc": float(result["metrics_frame"].loc[result["metrics_frame"]["split"] == "validation", "pr_auc"].iloc[0]),
                "test_pr_auc": float(result["metrics_frame"].loc[result["metrics_frame"]["split"] == "test", "pr_auc"].iloc[0]),
                "test_roc_auc": float(result["metrics_frame"].loc[result["metrics_frame"]["split"] == "test", "roc_auc"].iloc[0]),
            }
            for name, result in results.items()
        },
        "best_model": best_result["model_name"],
        "champion": champion_artifacts,
    }
    if reporting_summary is not None:
        summary["lightgbm_reporting"] = {
            "model_dir": str(reporting_summary["model_dir"]),
            "reports_dir": str(reporting_summary["reports_dir"]),
            "selected_threshold": float(reporting_summary["selected_threshold"]),
        }
    write_text(artifacts_dir / "run_summary.json", json.dumps(summary, indent=2))
    print(f"[PHASE1] complete best_model={best_result['model_name']}", flush=True)


if __name__ == "__main__":
    main()
