from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT_DIR / "runtime" / "response_modeling" / "models" / "lightgbm"
DEFAULT_REPORTS_DIR = ROOT_DIR / "runtime" / "reports"
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "response_model.yaml"
BUSINESS_EXCLUDED_SEGMENTS = {"UNKNOWN", "__NULL__"}
THRESHOLD_GRID = np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2)
MIN_THRESHOLD_GRID_ROWS = 1000
MIN_THRESHOLD_GRID_POSITIVES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild response-model VIP and threshold reporting from saved predictions.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--threshold-grid-min-rows", type=int, default=MIN_THRESHOLD_GRID_ROWS)
    parser.add_argument("--threshold-grid-min-positives", type=int, default=MIN_THRESHOLD_GRID_POSITIVES)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def normalize_segment_value(value: Any) -> str:
    return "__NULL__" if pd.isna(value) else str(value)


def format_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{int(value):,}"


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows_"
    view = frame.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    columns = [str(column) for column in view.columns]
    rows = []
    for _, row in view.iterrows():
        rows.append(["" if pd.isna(value) else str(value) for value in row.tolist()])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(values) + " |" for values in rows]
    return "\n".join([header, separator] + body)


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


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


def choose_threshold_from_grid(
    threshold_grid: pd.DataFrame,
    threshold_selection_config: Dict[str, Any] | None,
) -> tuple[float, str]:
    selection_config = threshold_selection_config or {}
    policy = str(selection_config.get("policy", "max_precision_with_min_recall")).strip().lower()
    fallback_policy = str(selection_config.get("fallback_policy", "max_precision")).strip().lower()
    minimum_recall = float(selection_config.get("minimum_recall", 0.0) or 0.0)
    frame = threshold_grid.sort_values("threshold").reset_index(drop=True)

    if policy == "max_precision_with_min_recall":
        eligible = frame.loc[frame["recall"] >= minimum_recall].copy()
        if not eligible.empty:
            selected = eligible.sort_values(
                ["precision", "f1", "recall", "threshold"],
                ascending=[False, False, False, False],
            ).iloc[0]
            return float(selected["threshold"]), f"max_precision_with_min_recall(minimum_recall={minimum_recall:.2f})"
        policy = fallback_policy

    if policy == "max_precision":
        selected = frame.sort_values(
            ["precision", "recall", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        return float(selected["threshold"]), "max_precision"

    if policy != "best_f1":
        raise ValueError(f"Unsupported threshold selection policy `{policy}`.")

    selected = frame.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return float(selected["threshold"]), "best_f1"


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    return classification_threshold_metrics(y_true, y_score, threshold)


def build_segment_metric_row(
    *,
    split_name: str,
    segment_value: str,
    frame: pd.DataFrame,
    threshold: float,
) -> Dict[str, Any]:
    y_true = frame["target"].to_numpy(dtype=int)
    y_score = frame["score"].to_numpy(dtype=float)
    row = {
        "rows": int(len(frame)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "score_mean": float(y_score.mean()),
        "pr_auc": safe_pr_auc(y_true, y_score),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "threshold": float(threshold),
        "row_count": int(len(frame)),
        "positive_count": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "model_name": "lightgbm",
        "split": split_name,
        "segment_column": "vip_level",
        "segment_value": segment_value,
    }
    row.update(threshold_metrics(y_true, y_score, threshold))
    return row


def build_vip_metrics(scored_by_split: Dict[str, pd.DataFrame], selected_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, frame in scored_by_split.items():
        for segment_value, segment_frame in frame.groupby("vip_level", dropna=False, observed=False):
            rows.append(
                build_segment_metric_row(
                    split_name=split_name,
                    segment_value=normalize_segment_value(segment_value),
                    frame=segment_frame.reset_index(drop=True),
                    threshold=selected_threshold,
                )
            )
    ordered_columns = [
        "rows",
        "positives",
        "positive_rate",
        "score_mean",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "predicted_positive_rate",
        "predicted_negative_rate",
        "positive_prediction_count",
        "negative_prediction_count",
        "specificity",
        "negative_predictive_value",
        "threshold",
        "row_count",
        "positive_count",
        "prevalence",
        "model_name",
        "split",
        "segment_column",
        "segment_value",
        "true_positive_count",
        "true_negative_count",
        "false_positive_count",
        "false_negative_count",
    ]
    return (
        pd.DataFrame(rows)[ordered_columns]
        .sort_values(["split", "row_count", "segment_value"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def build_threshold_grid_rows(
    *,
    split_name: str,
    segment_value: str,
    frame: pd.DataFrame,
    thresholds: Iterable[float],
) -> List[Dict[str, Any]]:
    y_true = frame["target"].to_numpy(dtype=int)
    y_score = frame["score"].to_numpy(dtype=float)
    pr_auc = safe_pr_auc(y_true, y_score)
    roc_auc = safe_roc_auc(y_true, y_score)
    row_count = int(len(frame))
    positive_count = int(y_true.sum())
    prevalence = float(y_true.mean())
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        metrics = threshold_metrics(y_true, y_score, float(threshold))
        rows.append(
            {
                "split": split_name,
                "segment_column": "vip_level",
                "segment_value": segment_value,
                "threshold": float(threshold),
                "row_count": row_count,
                "positive_count": positive_count,
                "prevalence": prevalence,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                **metrics,
            }
        )
    return rows


def build_global_threshold_grid(scored_by_split: Dict[str, pd.DataFrame], thresholds: Iterable[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, frame in scored_by_split.items():
        rows.extend(build_threshold_grid_rows(split_name=split_name, segment_value="GLOBAL", frame=frame, thresholds=thresholds))
    return pd.DataFrame(rows).sort_values(["split", "threshold"]).reset_index(drop=True)


def eligible_threshold_segments(
    frame: pd.DataFrame,
    *,
    min_rows: int,
    min_positives: int,
) -> pd.DataFrame:
    inventory = (
        frame.assign(segment_value=frame["vip_level"].map(normalize_segment_value))
        .groupby("segment_value", as_index=False, observed=False)
        .agg(row_count=("target", "size"), positive_count=("target", "sum"), prevalence=("target", "mean"))
        .sort_values(["row_count", "segment_value"], ascending=[False, True])
        .reset_index(drop=True)
    )
    inventory["eligible_for_threshold_grid"] = (inventory["row_count"] >= min_rows) & (inventory["positive_count"] >= min_positives)
    return inventory


def build_vip_threshold_grid(
    scored_by_split: Dict[str, pd.DataFrame],
    thresholds: Iterable[float],
    *,
    min_rows: int,
    min_positives: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    business_rows: List[Dict[str, Any]] = []
    technical_rows: List[Dict[str, Any]] = []
    inventory_rows: List[pd.DataFrame] = []

    for split_name, frame in scored_by_split.items():
        inventory = eligible_threshold_segments(frame, min_rows=min_rows, min_positives=min_positives)
        inventory.insert(0, "split", split_name)
        inventory_rows.append(inventory)
        eligible_segments = set(inventory.loc[inventory["eligible_for_threshold_grid"], "segment_value"])
        working = frame.assign(segment_value=frame["vip_level"].map(normalize_segment_value))
        for segment_value, segment_frame in working.groupby("segment_value", observed=False):
            if segment_value not in eligible_segments:
                continue
            rows = build_threshold_grid_rows(
                split_name=split_name,
                segment_value=segment_value,
                frame=segment_frame.reset_index(drop=True),
                thresholds=thresholds,
            )
            technical_rows.extend(rows)
            if segment_value not in BUSINESS_EXCLUDED_SEGMENTS:
                business_rows.extend(rows)

    technical_frame = pd.DataFrame(technical_rows).sort_values(["split", "segment_value", "threshold"]).reset_index(drop=True)
    business_frame = pd.DataFrame(business_rows).sort_values(["split", "segment_value", "threshold"]).reset_index(drop=True)
    inventory_frame = pd.concat(inventory_rows, ignore_index=True)
    return business_frame, technical_frame, inventory_frame


def read_selected_threshold(model_dir: Path) -> float:
    metrics = pd.read_csv(model_dir / "metrics.csv")
    threshold = pd.to_numeric(metrics["threshold"], errors="coerce").dropna()
    if threshold.empty:
        raise ValueError(f"Could not find a saved threshold in {(model_dir / 'metrics.csv')}.")
    return float(threshold.iloc[0])


def best_threshold_for_segment(
    threshold_grid: pd.DataFrame,
    *,
    split_name: str,
    segment_value: str,
) -> pd.Series:
    frame = threshold_grid.loc[
        (threshold_grid["split"] == split_name) & (threshold_grid["segment_value"] == segment_value)
    ].copy()
    return frame.sort_values(["f1", "precision", "recall", "threshold"], ascending=[False, False, False, True]).iloc[0]


def threshold_snapshot(
    threshold_grid: pd.DataFrame,
    *,
    split_name: str,
    segment_value: str,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    frame = threshold_grid.loc[
        (threshold_grid["split"] == split_name)
        & (threshold_grid["segment_value"] == segment_value)
        & (threshold_grid["threshold"].isin(list(thresholds)))
    ].copy()
    return frame.loc[
        :,
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "specificity",
            "negative_predictive_value",
            "positive_prediction_count",
            "negative_prediction_count",
            "predicted_positive_rate",
            "predicted_negative_rate",
        ],
    ].reset_index(drop=True)


def create_reports(
    *,
    reports_dir: Path,
    model_dir: Path,
    current_threshold: float,
    selected_threshold: float,
    threshold_policy: str,
    vip_metrics_business: pd.DataFrame,
    vip_metrics_technical: pd.DataFrame,
    threshold_grid_global: pd.DataFrame,
    threshold_grid_vip_business: pd.DataFrame,
    threshold_grid_vip_technical: pd.DataFrame,
    threshold_inventory: pd.DataFrame,
) -> None:
    test_business = vip_metrics_business.loc[vip_metrics_business["split"] == "test"].copy()
    validation_business = vip_metrics_business.loc[vip_metrics_business["split"] == "validation"].copy()
    test_technical = vip_metrics_technical.loc[vip_metrics_technical["split"] == "test"].copy()

    v1_test = test_technical.loc[test_technical["segment_value"] == "V1"].iloc[0]
    v2_test = test_technical.loc[test_technical["segment_value"] == "V2"].iloc[0]
    v1_validation = vip_metrics_technical.loc[
        (vip_metrics_technical["split"] == "validation") & (vip_metrics_technical["segment_value"] == "V1")
    ].iloc[0]
    v2_validation = vip_metrics_technical.loc[
        (vip_metrics_technical["split"] == "validation") & (vip_metrics_technical["segment_value"] == "V2")
    ].iloc[0]

    global_test = threshold_grid_global.loc[threshold_grid_global["split"] == "test"].copy()
    global_validation = threshold_grid_global.loc[threshold_grid_global["split"] == "validation"].copy()
    recall_view = global_test.loc[global_test["threshold"] == 0.30].iloc[0]
    balanced_view = global_test.loc[global_test["threshold"] == selected_threshold].iloc[0]
    precision_view = global_test.loc[global_test["threshold"] == 0.60].iloc[0]
    current_view_test = global_test.loc[global_test["threshold"] == current_threshold].iloc[0]
    current_view_validation = global_validation.loc[global_validation["threshold"] == current_threshold].iloc[0]
    recommended_view_validation = global_validation.loc[global_validation["threshold"] == selected_threshold].iloc[0]

    v1_best_test = best_threshold_for_segment(threshold_grid_vip_technical, split_name="test", segment_value="V1")
    v2_best_test = best_threshold_for_segment(threshold_grid_vip_technical, split_name="test", segment_value="V2")
    v1_best_validation = best_threshold_for_segment(threshold_grid_vip_technical, split_name="validation", segment_value="V1")
    v2_best_validation = best_threshold_for_segment(threshold_grid_vip_technical, split_name="validation", segment_value="V2")

    small_test_segments = threshold_inventory.loc[
        (threshold_inventory["split"] == "test") & (~threshold_inventory["eligible_for_threshold_grid"]),
        ["segment_value", "row_count", "positive_count", "prevalence"],
    ].copy()
    small_test_segments = small_test_segments.sort_values(["row_count", "segment_value"], ascending=[False, True]).reset_index(drop=True)

    v1_v2_summary = pd.DataFrame(
        [
            {
                "segment": "V1",
                "split": "validation",
                "row_count": int(v1_validation["row_count"]),
                "positive_count": int(v1_validation["positive_count"]),
                "prevalence": format_float(v1_validation["prevalence"]),
                "pr_auc": format_float(v1_validation["pr_auc"]),
                "roc_auc": format_float(v1_validation["roc_auc"]),
                "selected_threshold": format_float(v1_validation["threshold"], digits=2),
                "selected_f1": format_float(v1_validation["f1"]),
                "best_threshold_on_grid": format_float(v1_best_validation["threshold"], digits=2),
                "best_f1_on_grid": format_float(v1_best_validation["f1"]),
            },
            {
                "segment": "V1",
                "split": "test",
                "row_count": int(v1_test["row_count"]),
                "positive_count": int(v1_test["positive_count"]),
                "prevalence": format_float(v1_test["prevalence"]),
                "pr_auc": format_float(v1_test["pr_auc"]),
                "roc_auc": format_float(v1_test["roc_auc"]),
                "selected_threshold": format_float(v1_test["threshold"], digits=2),
                "selected_f1": format_float(v1_test["f1"]),
                "best_threshold_on_grid": format_float(v1_best_test["threshold"], digits=2),
                "best_f1_on_grid": format_float(v1_best_test["f1"]),
            },
            {
                "segment": "V2",
                "split": "validation",
                "row_count": int(v2_validation["row_count"]),
                "positive_count": int(v2_validation["positive_count"]),
                "prevalence": format_float(v2_validation["prevalence"]),
                "pr_auc": format_float(v2_validation["pr_auc"]),
                "roc_auc": format_float(v2_validation["roc_auc"]),
                "selected_threshold": format_float(v2_validation["threshold"], digits=2),
                "selected_f1": format_float(v2_validation["f1"]),
                "best_threshold_on_grid": format_float(v2_best_validation["threshold"], digits=2),
                "best_f1_on_grid": format_float(v2_best_validation["f1"]),
            },
            {
                "segment": "V2",
                "split": "test",
                "row_count": int(v2_test["row_count"]),
                "positive_count": int(v2_test["positive_count"]),
                "prevalence": format_float(v2_test["prevalence"]),
                "pr_auc": format_float(v2_test["pr_auc"]),
                "roc_auc": format_float(v2_test["roc_auc"]),
                "selected_threshold": format_float(v2_test["threshold"], digits=2),
                "selected_f1": format_float(v2_test["f1"]),
                "best_threshold_on_grid": format_float(v2_best_test["threshold"], digits=2),
                "best_f1_on_grid": format_float(v2_best_test["f1"]),
            },
        ]
    )

    reassessment_report = f"""
# Response Model VIP-Level Reassessment

This reassessment uses the saved LightGBM prediction artifacts in `{model_dir}`. No business target definition was changed. The target remains `response_label_positive_3d`.

## Test VIP Metrics For Business Segments

{markdown_table(test_business.loc[:, ["segment_value", "row_count", "positive_count", "prevalence", "pr_auc", "roc_auc", "precision", "recall", "f1", "predicted_positive_rate"]])}

## V1 And V2 Focus

{markdown_table(v1_v2_summary)}

## Commentary

- `V1` remains materially weak. Test PR-AUC is `{format_float(v1_test["pr_auc"])}` and ROC-AUC is `{format_float(v1_test["roc_auc"])}` on `{format_int(v1_test["row_count"])}` rows, so there is some ranking signal but not enough separation for a strong operating point.
- `V1` is underperforming for both reasons. The current global threshold `{format_float(selected_threshold, digits=2)}` is too conservative for this segment: test recall is only `{format_float(v1_test["recall"])}` with a predicted-positive rate of `{format_float(v1_test["predicted_positive_rate"])}` against prevalence `{format_float(v1_test["prevalence"])}`. Lowering the threshold to `{format_float(v1_best_test["threshold"], digits=2)}` improves test F1 from `{format_float(v1_test["f1"])}` to `{format_float(v1_best_test["f1"])}`, but the segment still only reaches precision `{format_float(v1_best_test["precision"])}` and recall `{format_float(v1_best_test["recall"])}`. That means threshold tuning helps, but weak ranking remains a real limitation.
- `V2` is better than `V1`, but still clearly below `V3`, `V4`, and `V5`. Test PR-AUC is `{format_float(v2_test["pr_auc"])}` and ROC-AUC is `{format_float(v2_test["roc_auc"])}` on `{format_int(v2_test["row_count"])}` rows.
- `V2` underperformance is mostly ranking-related rather than threshold-related. The current threshold `{format_float(selected_threshold, digits=2)}` gives test F1 `{format_float(v2_test["f1"])}`, while the best grid threshold `{format_float(v2_best_test["threshold"], digits=2)}` only nudges F1 to `{format_float(v2_best_test["f1"])}`. Threshold changes shift recall and volume, but do not fix the segment’s moderate ranking quality.
- `V1` and `V2` are both large enough to interpret. Their test row counts are `{format_int(v1_test["row_count"])}` and `{format_int(v2_test["row_count"])}` respectively, so their weak performance should be treated as real rather than as a pure small-sample artifact.
- `V6`, `V7`, and `V8` are too small for strong threshold conclusions in the current saved test slice. They have only `{", ".join(f"{row.segment_value}={format_int(row.row_count)} rows" for row in small_test_segments.itertuples() if row.segment_value in ['V6', 'V7', 'V8'])}`.

## Segments Too Small To Interpret Strongly

Threshold-grid sufficiency was defined explicitly as `row_count >= {MIN_THRESHOLD_GRID_ROWS}` and `positive_count >= {MIN_THRESHOLD_GRID_POSITIVES}` within a split. The following test segments did not meet that bar:

{markdown_table(small_test_segments)}

## Practical Next Recommendations

- Keep `V1` explicitly flagged as a weak segment in business-facing evaluation. Do not represent it as production-ready classification quality at the saved threshold.
- Keep `V2` visible as a middling segment: usable as part of a global ranking view, but materially weaker than `V3` to `V5`.
- Use the threshold grid when discussing operations. For `V1`, threshold choice changes the tradeoff materially. For `V2`, threshold choice mostly changes volume and recall, not underlying ranking quality.
- If the score is used operationally, keep one documented global threshold first and treat any later VIP-aware thresholding as a separate policy decision rather than a silent model-quality fix.
"""
    write_text(reports_dir / "response_model_vip_level_reassessment.md", reassessment_report)

    reporting_update_report = f"""
# Response Model VIP Reporting Update

The VIP reporting split is now explicit. Business-facing files exclude `NULL` and `UNKNOWN`. Technical files preserve them for debugging and data-quality review.

## Business-Facing Outputs

- `{model_dir / 'vip_level_metrics_validation.csv'}`
- `{model_dir / 'vip_level_metrics_test.csv'}`
- `{model_dir / 'threshold_grid_vip_level.csv'}`

These files only include named VIP business segments and exclude `UNKNOWN` plus physical nulls represented as `__NULL__`.

## Technical Outputs

- `{model_dir / 'vip_level_metrics_validation_technical.csv'}`
- `{model_dir / 'vip_level_metrics_test_technical.csv'}`
- `{model_dir / 'threshold_grid_vip_level_technical.csv'}`
- `{model_dir / 'segment_metrics.csv'}`

These files keep all persisted categories, including `UNKNOWN` and `__NULL__`, because they are still useful for model debugging, score-distribution review, and upstream data-quality checks.

## Where NULL And UNKNOWN Still Remain

- `UNKNOWN` remains in the technical VIP files because it is a real modeled category in the saved predictions.
- `__NULL__` remains in the technical VIP files because the saved prediction artifacts contain physical null `vip_level` values that are still important to audit.
- `threshold_grid_global.csv` remains global only and therefore has no VIP category rows.

## Current Business/Test VIP View

{markdown_table(test_business.loc[:, ["segment_value", "row_count", "positive_count", "prevalence", "pr_auc", "roc_auc", "precision", "recall", "f1"]])}
"""
    write_text(reports_dir / "response_model_vip_reporting_update.md", reporting_update_report)

    global_threshold_summary = global_test.loc[
        global_test["threshold"].isin([0.30, selected_threshold, 0.60]),
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "specificity",
            "negative_predictive_value",
            "positive_prediction_count",
            "negative_prediction_count",
            "predicted_positive_rate",
            "predicted_negative_rate",
            "row_count",
            "positive_count",
            "prevalence",
        ],
    ].reset_index(drop=True)

    threshold_grid_report = f"""
# Response Model Threshold Grid

This report uses the saved validation and test prediction parquets in `{model_dir}` and evaluates the explicit threshold grid `0.05, 0.10, ..., 0.95`.

## Output Files

- `{model_dir / 'threshold_grid_global.csv'}`
- `{model_dir / 'threshold_grid_vip_level.csv'}`
- `{model_dir / 'threshold_grid_vip_level_technical.csv'}`

## Threshold Grid Sufficiency Rule For VIP Segments

- A VIP segment is included in the threshold-grid VIP outputs only when `row_count >= {MIN_THRESHOLD_GRID_ROWS}` and `positive_count >= {MIN_THRESHOLD_GRID_POSITIVES}` in that split.
- This keeps the multi-threshold view readable and avoids over-interpreting tiny VIP segments.

## Global Test Threshold Snapshot

{markdown_table(global_threshold_summary)}

## Test Threshold Snapshot For V1

{markdown_table(threshold_snapshot(threshold_grid_vip_technical, split_name="test", segment_value="V1", thresholds=[0.25, 0.30, 0.35, 0.45, 0.60]))}

## Test Threshold Snapshot For V2

{markdown_table(threshold_snapshot(threshold_grid_vip_technical, split_name="test", segment_value="V2", thresholds=[0.25, 0.30, 0.35, 0.45, 0.60]))}

## Notes

- The current recommended threshold is `{format_float(selected_threshold, digits=2)}` under policy `{threshold_policy}`.
- `V1` shows a visibly different threshold preference from the global choice: F1 peaks lower, around `{format_float(v1_best_test["threshold"], digits=2)}` on test.
- `V2` is comparatively stable across `{format_float(0.30, digits=2)}` to `{format_float(0.45, digits=2)}`, which reinforces that its remaining gap is not mainly a threshold-selection problem.
"""
    write_text(reports_dir / "response_model_threshold_grid.md", threshold_grid_report)

    recommendation_rows = pd.DataFrame(
        [
            {
                "view": "recall-prioritized",
                "threshold": 0.30,
                "global_precision": recall_view["precision"],
                "global_recall": recall_view["recall"],
                "global_f1": recall_view["f1"],
                "global_predicted_positive_rate": recall_view["predicted_positive_rate"],
            },
            {
                "view": "balanced",
                "threshold": selected_threshold,
                "global_precision": balanced_view["precision"],
                "global_recall": balanced_view["recall"],
                "global_f1": balanced_view["f1"],
                "global_predicted_positive_rate": balanced_view["predicted_positive_rate"],
            },
            {
                "view": "precision-protected",
                "threshold": 0.60,
                "global_precision": precision_view["precision"],
                "global_recall": precision_view["recall"],
                "global_f1": precision_view["f1"],
                "global_predicted_positive_rate": precision_view["predicted_positive_rate"],
            },
        ]
    )

    threshold_recommendation_report = f"""
# Response Model Threshold Recommendation

The threshold grid does not justify hiding weak VIP segments. It does support a clearer operating-point discussion.

## Candidate Global Operating Views On Test

{markdown_table(recommendation_rows)}

## Interpretation

- `recall-prioritized` view at `0.30`: global recall rises to `{format_float(recall_view["recall"])}`, but predicted-positive rate expands to `{format_float(recall_view["predicted_positive_rate"])}`. For `V1`, this meaningfully improves recall to `{format_float(threshold_grid_vip_technical.loc[(threshold_grid_vip_technical['split'] == 'test') & (threshold_grid_vip_technical['segment_value'] == 'V1') & (threshold_grid_vip_technical['threshold'] == 0.30), 'recall'].iloc[0])}` and F1 to `{format_float(threshold_grid_vip_technical.loc[(threshold_grid_vip_technical['split'] == 'test') & (threshold_grid_vip_technical['segment_value'] == 'V1') & (threshold_grid_vip_technical['threshold'] == 0.30), 'f1'].iloc[0])}`, but precision stays modest at `{format_float(threshold_grid_vip_technical.loc[(threshold_grid_vip_technical['split'] == 'test') & (threshold_grid_vip_technical['segment_value'] == 'V1') & (threshold_grid_vip_technical['threshold'] == 0.30), 'precision'].iloc[0])}`.
- `balanced` view at `{format_float(selected_threshold, digits=2)}`: this is the recommended default because it was chosen on validation under the documented precision-with-recall-floor policy rather than on test. It gives global precision `{format_float(balanced_view["precision"])}` and recall `{format_float(balanced_view["recall"])}`. For `V2`, it stays close to the segment’s best observed F1 on the preferred grid.
- `precision-protected` view at `0.60`: global precision improves to `{format_float(precision_view["precision"])}`, while recall falls to `{format_float(precision_view["recall"])}`. This hurts `V1` sharply, dropping its recall to `{format_float(threshold_grid_vip_technical.loc[(threshold_grid_vip_technical['split'] == 'test') & (threshold_grid_vip_technical['segment_value'] == 'V1') & (threshold_grid_vip_technical['threshold'] == 0.60), 'recall'].iloc[0])}` and F1 to `{format_float(threshold_grid_vip_technical.loc[(threshold_grid_vip_technical['split'] == 'test') & (threshold_grid_vip_technical['segment_value'] == 'V1') & (threshold_grid_vip_technical['threshold'] == 0.60), 'f1'].iloc[0])}`.

## Recommendation

- Keep one documented global threshold as the default near the current validation-selected point `{format_float(selected_threshold, digits=2)}`. It is not segment-optimal, but it is transparent and avoids introducing silent policy complexity.
- Use the threshold grid to show that `V1` is the main exception. `V1` prefers a materially lower threshold than the global operating point, but even that lower threshold does not make the segment strong.
- Treat `V2` as acceptable for a global-threshold ranking view, but do not describe it as strong segment quality.
- VIP-aware thresholding can be considered later as an explicit policy-layer experiment. It should only be introduced after business review, because it changes who gets contacted by segment and should not be framed as a hidden metric fix.
"""
    write_text(reports_dir / "response_model_threshold_recommendation.md", threshold_recommendation_report)

    threshold_policy_review = f"""
# Response Threshold Policy Review

This review uses the saved LightGBM validation and test predictions in `{model_dir}`. The label definition is unchanged: `response_label_positive_3d` remains the positive observed 3-day response flag.

## Threshold Policy

- Current threshold before this refinement: `{format_float(current_threshold, digits=2)}`
- Previous selection basis: validation `best_f1`
- Recommended threshold: `{format_float(selected_threshold, digits=2)}`
- Recommended policy: `{threshold_policy}`

## Validation Threshold Comparison

{markdown_table(global_validation.loc[global_validation["threshold"].isin([current_threshold, selected_threshold, 0.60])].reset_index(drop=True))}

## Recommended Threshold Metrics On Test

{markdown_table(pd.DataFrame([balanced_view]).loc[:, ["threshold", "precision", "recall", "f1", "specificity", "negative_predictive_value", "positive_prediction_count", "negative_prediction_count", "predicted_positive_rate", "predicted_negative_rate"]])}

## Threshold Grid Summary

{markdown_table(global_validation.loc[:, ["threshold", "precision", "recall", "f1", "specificity", "negative_predictive_value", "positive_prediction_count", "negative_prediction_count", "predicted_positive_rate", "predicted_negative_rate"]])}

## Business Interpretation

- Moving from `{format_float(current_threshold, digits=2)}` to `{format_float(selected_threshold, digits=2)}` reduces broad targeting volume while lifting precision.
- Validation precision improves from `{format_float(current_view_validation["precision"])}` to `{format_float(recommended_view_validation["precision"])}` while recall stays at `{format_float(recommended_view_validation["recall"])}`, above the configured floor.
- On held-out test data, the recommended threshold targets `{format_int(balanced_view["positive_prediction_count"])}` of `{format_int(balanced_view["row_count"])}` users (`{format_float(balanced_view["predicted_positive_rate"])}`), leaving `{format_int(balanced_view["negative_prediction_count"])}` users untouched by default.
"""
    write_text(reports_dir / "response_threshold_policy_review.md", threshold_policy_review)

    metric_extension_summary = f"""
# Response Metric Extension Summary

The response-model evaluation outputs now include the additional thresholded classification metrics requested for deployment review.

## Added Metrics

- `positive_prediction_count`
- `negative_prediction_count`
- `predicted_positive_rate`
- `predicted_negative_rate`
- `specificity`
- `negative_predictive_value`

## Where They Now Appear

- `{model_dir / 'metrics.csv'}` for global train, validation, and test metrics at the selected threshold
- `{model_dir / 'segment_metrics.csv'}` for thresholded segment metrics
- `{model_dir / 'vip_level_metrics_validation.csv'}`
- `{model_dir / 'vip_level_metrics_test.csv'}`
- `{model_dir / 'threshold_table.csv'}`
- `{model_dir / 'threshold_grid_global.csv'}`
- `{model_dir / 'threshold_grid_vip_level.csv'}`
- `{model_dir / 'threshold_grid_vip_level_technical.csv'}`

## Global Selected-Threshold Metrics

{markdown_table(pd.DataFrame([balanced_view]).loc[:, ["threshold", "precision", "recall", "f1", "specificity", "negative_predictive_value", "positive_prediction_count", "negative_prediction_count", "predicted_positive_rate", "predicted_negative_rate"]])}

## VIP-Level Support

- Global metrics are emitted for train, validation, and test.
- `vip_level` metrics are emitted where VIP reporting is already supported.
- Threshold-grid VIP outputs still respect the existing sample-size sufficiency rule before showing a segment.

## MLflow

- No active MLflow logging path is implemented in this response pipeline codebase today, so the extension was applied to persisted CSV reporting and markdown outputs only.
"""
    write_text(reports_dir / "response_metric_extension_summary.md", metric_extension_summary)

    business_threshold_report = f"""
# Response Model Business Threshold Report

## What The Model Predicts

- The model predicts the probability of a positive observed 3-day response after historical treatment patterns similar to the training data.
- `label = 1` means a positive response was observed.
- `label = 0` means no positive response was observed in the response window. It does not mean harm.
- The prediction score is the estimated probability of positive response, not proof of causal uplift.

## Recommended Operating Point

- Recommended threshold: `{format_float(selected_threshold, digits=2)}`
- Users predicted positive on held-out test data: `{format_int(balanced_view["positive_prediction_count"])}` (`{format_float(balanced_view["predicted_positive_rate"])}`)
- Users predicted negative on held-out test data: `{format_int(balanced_view["negative_prediction_count"])}` (`{format_float(balanced_view["predicted_negative_rate"])}`)

## Operational Meaning

- Users above the threshold should be considered for treatment within already-approved campaign pools because they are more likely to show a positive observed response.
- Users below the threshold should generally not be touched for broad treatment pushes and can be deprioritized when budget or contact capacity is limited.

## Precision And Recall Trade-Off

- At `{format_float(current_threshold, digits=2)}`, the old operating point delivered test precision `{format_float(current_view_test["precision"])}` and recall `{format_float(current_view_test["recall"])}`.
- At `{format_float(selected_threshold, digits=2)}`, the recommended operating point improves test precision to `{format_float(balanced_view["precision"])}` while recall moves to `{format_float(balanced_view["recall"])}`.
- At `0.60`, precision rises further to `{format_float(precision_view["precision"])}`, but recall drops to `{format_float(precision_view["recall"])}`, which is likely too restrictive for default use.

## Caveat

- This is observational response modeling. It helps rank likely responders among already-eligible users, but it does not prove incremental uplift or causal treatment effect.
"""
    write_text(reports_dir / "response_model_business_threshold_report.md", business_threshold_report)


def generate_reporting_artifacts(
    *,
    model_dir: Path,
    reports_dir: Path,
    scored_by_split: Dict[str, pd.DataFrame] | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    threshold_grid_min_rows: int = MIN_THRESHOLD_GRID_ROWS,
    threshold_grid_min_positives: int = MIN_THRESHOLD_GRID_POSITIVES,
) -> Dict[str, Any]:
    model_dir = model_dir.resolve()
    reports_dir = reports_dir.resolve()
    config_path = config_path.resolve()
    print(f"[PHASE1] using_config_path={config_path}")
    ensure_dir(reports_dir)

    if scored_by_split is None:
        raise ValueError("generate_reporting_artifacts requires in-memory scored_by_split frames in the canonical runtime.")
    config = load_yaml(config_path)
    threshold_grid_global = build_global_threshold_grid(scored_by_split, THRESHOLD_GRID)
    current_threshold, _ = choose_threshold_from_grid(
        threshold_grid_global.loc[threshold_grid_global["split"] == "validation"].copy(),
        {"policy": "best_f1"},
    )
    selected_threshold, threshold_policy = choose_threshold_from_grid(
        threshold_grid_global.loc[threshold_grid_global["split"] == "validation"].copy(),
        dict(config.get("threshold_selection", {})),
    )

    vip_metrics_technical = build_vip_metrics(scored_by_split, selected_threshold)
    vip_metrics_business = vip_metrics_technical.loc[~vip_metrics_technical["segment_value"].isin(BUSINESS_EXCLUDED_SEGMENTS)].copy()

    save_frame(model_dir / "vip_level_metrics_validation.csv", vip_metrics_business.loc[vip_metrics_business["split"] == "validation"])
    save_frame(model_dir / "vip_level_metrics_test.csv", vip_metrics_business.loc[vip_metrics_business["split"] == "test"])
    save_frame(model_dir / "vip_level_metrics_validation_technical.csv", vip_metrics_technical.loc[vip_metrics_technical["split"] == "validation"])
    save_frame(model_dir / "vip_level_metrics_test_technical.csv", vip_metrics_technical.loc[vip_metrics_technical["split"] == "test"])

    threshold_grid_vip_business, threshold_grid_vip_technical, threshold_inventory = build_vip_threshold_grid(
        scored_by_split,
        THRESHOLD_GRID,
        min_rows=int(threshold_grid_min_rows),
        min_positives=int(threshold_grid_min_positives),
    )

    save_frame(model_dir / "threshold_grid_global.csv", threshold_grid_global)
    save_frame(model_dir / "threshold_grid_vip_level.csv", threshold_grid_vip_business)
    save_frame(model_dir / "threshold_grid_vip_level_technical.csv", threshold_grid_vip_technical)
    save_frame(model_dir / "threshold_grid_vip_level_segment_inventory.csv", threshold_inventory)

    create_reports(
        reports_dir=reports_dir,
        model_dir=model_dir,
        current_threshold=current_threshold,
        selected_threshold=selected_threshold,
        threshold_policy=threshold_policy,
        vip_metrics_business=vip_metrics_business,
        vip_metrics_technical=vip_metrics_technical,
        threshold_grid_global=threshold_grid_global,
        threshold_grid_vip_business=threshold_grid_vip_business,
        threshold_grid_vip_technical=threshold_grid_vip_technical,
        threshold_inventory=threshold_inventory,
    )

    return {
        "model_dir": model_dir,
        "reports_dir": reports_dir,
        "config_path": config_path,
        "current_threshold": current_threshold,
        "selected_threshold": selected_threshold,
        "threshold_policy": threshold_policy,
        "vip_metrics_business": vip_metrics_business,
        "vip_metrics_technical": vip_metrics_technical,
        "threshold_grid_global": threshold_grid_global,
        "threshold_grid_vip_business": threshold_grid_vip_business,
        "threshold_grid_vip_technical": threshold_grid_vip_technical,
        "threshold_inventory": threshold_inventory,
    }


def main() -> None:
    args = parse_args()
    result = generate_reporting_artifacts(
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        config_path=args.config_path,
        threshold_grid_min_rows=int(args.threshold_grid_min_rows),
        threshold_grid_min_positives=int(args.threshold_grid_min_positives),
    )

    print(f"[REPORTING] model_dir={result['model_dir']}")
    print(f"[REPORTING] current_threshold={result['current_threshold']:.2f}")
    print(f"[REPORTING] selected_threshold={result['selected_threshold']:.2f}")
    print(f"[REPORTING] threshold_policy={result['threshold_policy']}")
    print(f"[REPORTING] reports_dir={result['reports_dir']}")


if __name__ == "__main__":
    main()
