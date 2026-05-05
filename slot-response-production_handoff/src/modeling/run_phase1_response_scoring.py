from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from data.odps_reader import create_odps_client_from_env, fetch_sql_as_frame, list_odps_partition_values  # noqa: E402
    from data.odps_writer import write_frame_to_odps  # noqa: E402
    from modeling.inference_feature_contract import (  # noqa: E402
        build_named_transformed_frame,
        format_feature_preview,
        resolve_expected_raw_feature_order,
        resolve_expected_transformed_feature_names,
        validate_scoring_input_frame,
    )
    from modeling.production_model_registry import DEFAULT_CHAMPION_REFERENCE_PATH, resolve_production_model_reference  # noqa: E402
else:  # pragma: no cover
    from ..data.odps_reader import create_odps_client_from_env, fetch_sql_as_frame, list_odps_partition_values
    from ..data.odps_writer import write_frame_to_odps
    from .inference_feature_contract import (
        build_named_transformed_frame,
        format_feature_preview,
        resolve_expected_raw_feature_order,
        resolve_expected_transformed_feature_names,
        validate_scoring_input_frame,
    )
    from .production_model_registry import DEFAULT_CHAMPION_REFERENCE_PATH, resolve_production_model_reference


REQUIRED_OUTPUT_COLUMNS = [
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
CRITICAL_NON_NULL_COLUMNS = REQUIRED_OUTPUT_COLUMNS[:-1]
REQUIRED_ODPS_ENV_VARS = (
    "ALIBABA_CLOUD_ACCESS_KEY_ID",
    "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
    "ODPS_PROJECT",
    "ODPS_ENDPOINT",
)


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def bootstrap_runtime_environment() -> None:
    mpl_config_dir = (PROJECT_ROOT / ".mplconfig").resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    missing_odps_vars = [name for name in REQUIRED_ODPS_ENV_VARS if not os.getenv(name, "").strip()]
    if missing_odps_vars:
        raise ValueError(
            "Missing required ODPS environment variables: "
            + ", ".join(missing_odps_vars)
        )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def format_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{int(value):,}"


def parse_partition_value(spec: Any) -> str:
    text = str(spec)
    start = text.find("'")
    end = text.rfind("'")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not parse partition spec: {spec}")
    return text[start + 1 : end]


def list_partitions(table_name: str) -> List[str]:
    return list_odps_partition_values(table_name, partition_column="pt")


def fetch_frame(sql: str, batch_size: int = 250000) -> pd.DataFrame:
    client = create_odps_client_from_env()
    return fetch_sql_as_frame(sql, odps_client=client, batch_size=batch_size)


def build_select_sql(table_name: str, columns: Iterable[str], where_clauses: Iterable[str]) -> str:
    return f"select {', '.join(columns)} from {table_name} where {' and '.join(where_clauses)}"


def compute_missingness(frame: pd.DataFrame) -> pd.DataFrame:
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


def build_score_distribution(scores: pd.Series) -> pd.DataFrame:
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    quantile_rows = [
        {"metric": f"p{int(q * 100):02d}" if q not in {0.0, 1.0} else ("min" if q == 0.0 else "max"), "value": float(scores.quantile(q))}
        for q in quantiles
    ]
    other_rows = [
        {"metric": "mean", "value": float(scores.mean())},
        {"metric": "std", "value": float(scores.std())},
    ]
    return pd.DataFrame(quantile_rows + other_rows)


def assign_priority_buckets(scores: pd.Series, bucket_labels: List[str]) -> tuple[pd.Series, Dict[str, float]]:
    if len(bucket_labels) != 5:
        raise ValueError("This scoring workflow currently expects exactly 5 priority bucket labels.")
    q20 = float(scores.quantile(0.2))
    q40 = float(scores.quantile(0.4))
    q60 = float(scores.quantile(0.6))
    q80 = float(scores.quantile(0.8))
    thresholds = {
        bucket_labels[0]: float("-inf"),
        bucket_labels[1]: q20,
        bucket_labels[2]: q40,
        bucket_labels[3]: q60,
        bucket_labels[4]: q80,
    }

    def _label(score: float) -> str:
        if score >= q80:
            return bucket_labels[4]
        if score >= q60:
            return bucket_labels[3]
        if score >= q40:
            return bucket_labels[2]
        if score >= q20:
            return bucket_labels[1]
        return bucket_labels[0]

    return scores.map(_label), thresholds


def action_recommendation(bucket: str, vip_level: Any, rules: Dict[str, Any]) -> str:
    high_priority = set(rules["high_priority_buckets"])
    low_priority = set(rules["low_priority_buckets"])
    high_value_vips = set(rules["high_value_vip_levels"])
    vip = None if pd.isna(vip_level) else str(vip_level)
    if bucket in high_priority:
        return "prioritize_for_campaign_review"
    if bucket == "medium" and vip in high_value_vips:
        return "manual_review_high_value_medium_score"
    if bucket in low_priority:
        return "deprioritize_aggressive_promo"
    return "standard_priority_review"


def derive_model_version(model_reference_path: Path, strategy: str) -> str:
    normalized = str(strategy).strip().lower()
    if normalized != "sha256":
        raise ValueError(f"Unsupported model_version_strategy `{strategy}`. Expected `sha256`.")
    digest = hashlib.sha256(model_reference_path.read_bytes()).hexdigest()[:16]
    return f"sha256:{digest}"


def resolve_cli_flag(cli_value: Optional[bool], config_value: Any) -> bool:
    if cli_value is None:
        return bool(config_value)
    return bool(cli_value)


def resolve_champion_reference_path(config_value: Optional[str]) -> Path:
    if config_value:
        candidate = Path(str(config_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate
    return (PROJECT_ROOT / DEFAULT_CHAMPION_REFERENCE_PATH).resolve()


def resolve_scoring_partition(feature_table: str, explicit_scoring_pt: Optional[str]) -> str:
    partitions = list_partitions(feature_table)
    if not partitions:
        raise ValueError(f"No partitions found in feature source table `{feature_table}`.")
    if explicit_scoring_pt:
        if explicit_scoring_pt not in partitions:
            raise ValueError(
                f"Requested scoring_pt `{explicit_scoring_pt}` does not exist in `{feature_table}`. "
                f"Latest available partition is `{partitions[-1]}`."
            )
        return explicit_scoring_pt
    return partitions[-1]


def coerce_numeric_features(frame: pd.DataFrame, numeric_feature_columns: Sequence[str]) -> pd.DataFrame:
    converted = frame.copy()
    for column in numeric_feature_columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def transform_scoring_input(
    *,
    scoring_input: pd.DataFrame,
    preprocessor: Any,
    transformed_feature_names: Sequence[str],
) -> pd.DataFrame:
    transformed = preprocessor.transform(scoring_input)
    return build_named_transformed_frame(
        transformed,
        transformed_feature_names=transformed_feature_names,
        index=scoring_input.index,
    )


def validate_output_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_OUTPUT_COLUMNS if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in REQUIRED_OUTPUT_COLUMNS]
    if missing or unexpected:
        raise ValueError(
            "Scoring output schema does not match the publish contract. "
            f"missing={missing}, unexpected={unexpected}"
        )


def validate_prewrite_output(
    *,
    frame: pd.DataFrame,
    scoring_pt: str,
    allowed_bucket_labels: Sequence[str],
) -> Dict[str, Any]:
    validate_output_columns(frame)
    errors: List[str] = []
    score_date_expected = datetime.strptime(scoring_pt, "%Y%m%d").strftime("%Y-%m-%d")
    duplicate_player_id_rows = int(frame.duplicated(subset=["player_id"], keep=False).sum())
    duplicate_partition_rows = int(frame.duplicated(subset=["player_id", "pt"], keep=False).sum())
    invalid_bucket_rows = int((~frame["response_priority_bucket"].astype(str).isin(list(allowed_bucket_labels))).sum())
    null_counts = {column: int(frame[column].isna().sum()) for column in CRITICAL_NON_NULL_COLUMNS}
    invalid_score_rows = int(
        frame["predicted_response_score"].isna().sum()
        + ((frame["predicted_response_score"] < 0.0) | (frame["predicted_response_score"] > 1.0)).sum()
    )
    distinct_scoring_pts = sorted(frame["scoring_pt"].dropna().astype(str).unique().tolist())
    distinct_partition_values = sorted(frame["pt"].dropna().astype(str).unique().tolist())
    distinct_score_dates = sorted(frame["score_date"].dropna().astype(str).unique().tolist())

    if frame.empty:
        errors.append("Scored dataset is empty.")
    if duplicate_player_id_rows:
        errors.append(f"Detected {duplicate_player_id_rows} duplicate player_id rows before publish.")
    if duplicate_partition_rows:
        errors.append(f"Detected {duplicate_partition_rows} duplicate player_id+pt rows before publish.")
    if invalid_score_rows:
        errors.append(f"Detected {invalid_score_rows} rows with invalid predicted_response_score values.")
    if invalid_bucket_rows:
        errors.append(f"Detected {invalid_bucket_rows} rows with invalid response_priority_bucket values.")
    for column, null_count in null_counts.items():
        if null_count:
            errors.append(f"Critical output column `{column}` contains {null_count} null rows.")
    if distinct_scoring_pts != [scoring_pt]:
        errors.append(f"`scoring_pt` must be exactly `{scoring_pt}` but was {distinct_scoring_pts}.")
    if distinct_partition_values != [scoring_pt]:
        errors.append(f"`pt` must be exactly `{scoring_pt}` but was {distinct_partition_values}.")
    if distinct_score_dates != [score_date_expected]:
        errors.append(f"`score_date` must be exactly `{score_date_expected}` but was {distinct_score_dates}.")

    if errors:
        raise ValueError("Pre-write validation failed:\n- " + "\n- ".join(errors))

    return {
        "row_count": int(len(frame)),
        "duplicate_player_id_rows": duplicate_player_id_rows,
        "duplicate_partition_rows": duplicate_partition_rows,
        "invalid_score_rows": invalid_score_rows,
        "invalid_bucket_rows": invalid_bucket_rows,
        "null_counts": null_counts,
        "scoring_pt": scoring_pt,
        "score_date": score_date_expected,
    }


def fetch_target_partition_frame(
    *,
    target_table: str,
    partition_column: str,
    scoring_pt: str,
) -> pd.DataFrame:
    sql = build_select_sql(
        target_table,
        REQUIRED_OUTPUT_COLUMNS[:-1] + [partition_column],
        [f"{partition_column} = '{scoring_pt}'"],
    )
    return fetch_frame(sql, batch_size=250000)


def fetch_target_partition_row_count(
    *,
    target_table: str,
    partition_column: str,
    scoring_pt: str,
) -> int:
    sql = (
        f"select count(1) as row_count from {target_table} "
        f"where {partition_column} = '{scoring_pt}'"
    )
    frame = fetch_frame(sql, batch_size=10000)
    if frame.empty:
        return 0
    return int(frame.iloc[0]["row_count"])


def validate_postwrite_output(
    *,
    written_frame: pd.DataFrame,
    expected_frame: pd.DataFrame,
    scoring_pt: str,
    allowed_bucket_labels: Sequence[str],
    expected_model_metadata: Dict[str, Any],
    partition_column: str,
) -> Dict[str, Any]:
    validate_output_columns(written_frame.rename(columns={partition_column: "pt"}))
    if len(written_frame) != len(expected_frame):
        raise ValueError(
            "Post-write validation failed: row count mismatch between scored output and written ODPS partition. "
            f"expected={len(expected_frame)}, actual={len(written_frame)}"
        )

    duplicate_player_id_rows = int(written_frame.duplicated(subset=["player_id"], keep=False).sum())
    if duplicate_player_id_rows:
        raise ValueError(f"Post-write validation failed: detected {duplicate_player_id_rows} duplicate player_id rows.")

    null_counts = {column: int(written_frame[column].isna().sum()) for column in CRITICAL_NON_NULL_COLUMNS}
    bad_nulls = {column: count for column, count in null_counts.items() if count}
    if bad_nulls:
        raise ValueError(f"Post-write validation failed: critical nulls detected {bad_nulls}.")

    invalid_score_rows = int(
        written_frame["predicted_response_score"].isna().sum()
        + ((written_frame["predicted_response_score"] < 0.0) | (written_frame["predicted_response_score"] > 1.0)).sum()
    )
    if invalid_score_rows:
        raise ValueError(f"Post-write validation failed: detected {invalid_score_rows} out-of-range scores.")

    invalid_bucket_rows = int((~written_frame["response_priority_bucket"].astype(str).isin(list(allowed_bucket_labels))).sum())
    if invalid_bucket_rows:
        raise ValueError(f"Post-write validation failed: detected {invalid_bucket_rows} invalid bucket labels.")

    distinct_partition_values = sorted(written_frame[partition_column].dropna().astype(str).unique().tolist())
    distinct_scoring_pts = sorted(written_frame["scoring_pt"].dropna().astype(str).unique().tolist())
    if distinct_partition_values != [scoring_pt]:
        raise ValueError(f"Post-write validation failed: partition values were {distinct_partition_values}, expected `{scoring_pt}`.")
    if distinct_scoring_pts != [scoring_pt]:
        raise ValueError(f"Post-write validation failed: scoring_pt values were {distinct_scoring_pts}, expected `{scoring_pt}`.")

    for column, expected_value in expected_model_metadata.items():
        actual_values = sorted(written_frame[column].dropna().astype(str).unique().tolist())
        if actual_values != [str(expected_value)]:
            raise ValueError(
                "Post-write validation failed: model metadata mismatch. "
                f"column={column}, expected={[str(expected_value)]}, actual={actual_values}"
            )

    return {
        "row_count": int(len(written_frame)),
        "duplicate_player_id_rows": duplicate_player_id_rows,
        "null_counts": null_counts,
        "invalid_score_rows": invalid_score_rows,
        "invalid_bucket_rows": invalid_bucket_rows,
        "distinct_partition_values": distinct_partition_values,
        "distinct_scoring_pts": distinct_scoring_pts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the current eligible audience and publish directly to the final Phase-1 response ODPS table.")
    parser.add_argument("--config", "--config-path", dest="config_path", default=str(PROJECT_ROOT / "configs" / "response_scoring.yaml"))
    parser.add_argument("--pt", "--scoring-pt", dest="scoring_pt", default=None)
    parser.add_argument("--output-target", default=None)
    parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    bootstrap_runtime_environment()

    scoring_config = load_yaml(Path(args.config_path).resolve())
    training_config = load_yaml((PROJECT_ROOT / scoring_config["training_feature_config"]).resolve())
    publish_config = dict(scoring_config["publish"])
    publish_enabled = resolve_cli_flag(args.publish, publish_config["enabled"])
    dry_run = resolve_cli_flag(args.dry_run, publish_config.get("dry_run", False))
    target_table = str(args.output_target or publish_config["target_table"])
    partition_column = str(publish_config["partition_column"])
    write_mode = str(publish_config["write_mode"]).strip().lower()
    if write_mode != "overwrite":
        raise ValueError(f"Unsafe publish mode `{write_mode}` configured. Production scoring only supports `overwrite`.")

    champion_reference_path = resolve_champion_reference_path(scoring_config.get("champion_reference_path"))
    champion_reference = resolve_production_model_reference(
        champion_reference_path=champion_reference_path,
    )
    model_name = str(champion_reference["model_name"])
    model_version = str(champion_reference["model_version"])
    model_reference_path = champion_reference_path
    model_artifact_path = Path(str(champion_reference["model_dir"])).resolve() / "model.joblib"
    model_payload = joblib.load(model_artifact_path)
    preprocessor = model_payload["preprocessor"]
    estimator = model_payload["estimator"]
    selected_threshold = float(champion_reference["selected_threshold"])
    model_timestamp = datetime.fromtimestamp(model_artifact_path.stat().st_mtime).isoformat()
    print(
        f"[SCORING] model_name={model_name} "
        f"model_version={model_version} "
        f"model_reference_path={model_reference_path} "
        f"model_artifact_path={model_artifact_path} "
        f"model_timestamp={model_timestamp}",
        flush=True,
    )

    feature_table = str(scoring_config["feature_source_table"])
    scoring_pt = resolve_scoring_partition(feature_table, args.scoring_pt)
    score_date = datetime.strptime(scoring_pt, "%Y%m%d").strftime("%Y-%m-%d")
    scoring_ts = datetime.utcnow().replace(microsecond=0)

    expected_raw_feature_order = resolve_expected_raw_feature_order(
        declared_feature_columns=list(training_config["numeric_feature_columns"]) + list(training_config["categorical_feature_columns"]),
        preprocessor=preprocessor,
    )
    transformed_feature_names = resolve_expected_transformed_feature_names(
        declared_transformed_feature_names=model_payload.get("feature_names", []),
        preprocessor=preprocessor,
        estimator=estimator,
    )
    raw_preview = format_feature_preview(expected_raw_feature_order)
    transformed_preview = format_feature_preview(transformed_feature_names)
    print(
        f"[FEATURES] raw_feature_count={raw_preview['feature_count']} "
        f"raw_feature_preview={raw_preview['feature_preview']}",
        flush=True,
    )
    print(
        f"[FEATURES] transformed_feature_count={transformed_preview['feature_count']} "
        f"transformed_feature_preview={transformed_preview['feature_preview']}",
        flush=True,
    )

    select_columns = list(dict.fromkeys(scoring_config["business_columns"] + ["is_eligible"] + expected_raw_feature_order))
    sql = build_select_sql(
        feature_table,
        select_columns,
        [
            f"{scoring_config['feature_partition_column']} = '{scoring_pt}'",
            "is_eligible = 1",
        ],
    )
    print(f"[SCORING] loading current eligible audience pt={scoring_pt}", flush=True)
    current_population = fetch_frame(sql, batch_size=250000)
    print(f"[SCORING] loaded_rows={len(current_population)}", flush=True)

    missing_required = [column for column in scoring_config["business_columns"] + expected_raw_feature_order if column not in current_population.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in scoring population: {missing_required}")

    current_population = current_population.loc[current_population["player_id"].notna()].copy()
    current_population = coerce_numeric_features(current_population, training_config["numeric_feature_columns"])
    scoring_input = current_population.loc[:, expected_raw_feature_order].copy()
    validate_scoring_input_frame(
        scoring_input,
        expected_raw_feature_order=expected_raw_feature_order,
    )
    transformed_frame = transform_scoring_input(
        scoring_input=scoring_input,
        preprocessor=preprocessor,
        transformed_feature_names=transformed_feature_names,
    )
    scores = estimator.predict_proba(transformed_frame)[:, 1]

    output_frame = current_population.loc[:, ["player_id", "snapshot_date", "vip_level"]].copy()
    output_frame["score_date"] = score_date
    output_frame["scoring_pt"] = scoring_pt
    output_frame["scoring_ts"] = scoring_ts
    output_frame["predicted_response_score"] = scores
    output_frame = output_frame.sort_values(["predicted_response_score", "player_id"], ascending=[False, True]).reset_index(drop=True)
    output_frame["score_rank"] = np.arange(1, len(output_frame) + 1)
    output_frame["score_percentile"] = 1.0 - ((output_frame["score_rank"] - 1) / len(output_frame))

    bucket_labels = list(scoring_config["prioritization"]["bucket_labels"])
    bucket_series, thresholds = assign_priority_buckets(output_frame["predicted_response_score"], bucket_labels)
    output_frame["response_priority_bucket"] = bucket_series.astype(str)
    output_frame["action_recommendation"] = output_frame.apply(
        lambda row: action_recommendation(str(row["response_priority_bucket"]), row["vip_level"], scoring_config["action_rules"]),
        axis=1,
    )
    output_frame["model_name"] = model_name
    output_frame["model_version"] = model_version
    output_frame["model_reference_path"] = str(model_reference_path)
    output_frame["selected_threshold"] = selected_threshold
    output_frame["pt"] = scoring_pt
    output_frame = output_frame.loc[:, REQUIRED_OUTPUT_COLUMNS]
    predicted_positive_mask = output_frame["predicted_response_score"] >= selected_threshold
    positive_prediction_count = int(predicted_positive_mask.sum())
    negative_prediction_count = int((~predicted_positive_mask).sum())
    predicted_positive_rate = float(predicted_positive_mask.mean()) if len(output_frame) else 0.0
    predicted_negative_rate = float(1.0 - predicted_positive_rate) if len(output_frame) else 0.0

    prewrite_summary = validate_prewrite_output(
        frame=output_frame,
        scoring_pt=scoring_pt,
        allowed_bucket_labels=bucket_labels,
    )

    score_distribution = build_score_distribution(output_frame["predicted_response_score"])
    bucket_summary = (
        output_frame.groupby("response_priority_bucket", as_index=False, observed=True)
        .agg(
            row_count=("player_id", "size"),
            min_score=("predicted_response_score", "min"),
            max_score=("predicted_response_score", "max"),
        )
        .sort_values("response_priority_bucket")
        .reset_index(drop=True)
    )
    threshold_rules = {
        "very_low": f"score < {thresholds['low']:.6f}",
        "low": f"{thresholds['low']:.6f} <= score < {thresholds['medium']:.6f}",
        "medium": f"{thresholds['medium']:.6f} <= score < {thresholds['high']:.6f}",
        "high": f"{thresholds['high']:.6f} <= score < {thresholds['very_high']:.6f}",
        "very_high": f"score >= {thresholds['very_high']:.6f}",
    }
    bucket_summary["population_share"] = bucket_summary["row_count"] / len(output_frame)
    bucket_summary["threshold_rule"] = bucket_summary["response_priority_bucket"].map(threshold_rules)
    bucket_vip_mix = (
        output_frame.assign(vip_level=output_frame["vip_level"].fillna("__NULL__").astype(str))
        .groupby(["response_priority_bucket", "vip_level"], as_index=False, observed=True)
        .agg(row_count=("player_id", "size"))
        .sort_values(["response_priority_bucket", "row_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    bucket_totals = bucket_vip_mix.groupby("response_priority_bucket", observed=True)["row_count"].transform("sum")
    bucket_vip_mix["bucket_share"] = bucket_vip_mix["row_count"] / bucket_totals

    before_write_count = fetch_target_partition_row_count(
        target_table=target_table,
        partition_column=partition_column,
        scoring_pt=scoring_pt,
    )
    print(
        "[SCORING] target_table="
        f"{target_table} partition={partition_column}='{scoring_pt}' row_count_before_write={before_write_count}",
        flush=True,
    )

    if dry_run:
        print("[SCORING] dry_run=true; skipping ODPS publish after successful pre-write validation.", flush=True)
    elif publish_enabled:
        print(
            f"[SCORING] publishing directly to {target_table} partition {partition_column}='{scoring_pt}' "
            f"using overwrite mode",
            flush=True,
        )
        write_frame_to_odps(
            output_frame,
            odps_target=target_table,
            partition_column=partition_column,
            partition_value=scoring_pt,
            write_mode="overwrite",
        )
    else:
        print("[SCORING] publish=false; ODPS publish skipped after successful pre-write validation.", flush=True)

    after_write_count = before_write_count
    postwrite_summary: Dict[str, Any] | None = None
    if not dry_run and publish_enabled:
        after_write_count = fetch_target_partition_row_count(
            target_table=target_table,
            partition_column=partition_column,
            scoring_pt=scoring_pt,
        )
        written_frame = fetch_target_partition_frame(
            target_table=target_table,
            partition_column=partition_column,
            scoring_pt=scoring_pt,
        )
        postwrite_summary = validate_postwrite_output(
            written_frame=written_frame,
            expected_frame=output_frame,
            scoring_pt=scoring_pt,
            allowed_bucket_labels=bucket_labels,
            expected_model_metadata={
                "model_name": model_name,
                "model_version": model_version,
                "model_reference_path": str(model_reference_path),
                "selected_threshold": selected_threshold,
            },
            partition_column=partition_column,
        )
        print(
            "[SCORING] post_write_validation_passed "
            f"row_count_after_write={after_write_count} duplicate_player_id_rows=0",
            flush=True,
        )

    run_summary = {
        "as_of_date": scoring_config.get("as_of_date"),
        "scoring_pt": scoring_pt,
        "score_date": score_date,
        "scoring_ts": scoring_ts,
        "feature_source_table": feature_table,
        "scoring_population_size": int(len(output_frame)),
        "model_name": model_name,
        "model_version": model_version,
        "model_reference_path": str(model_reference_path),
        "selected_threshold": selected_threshold,
        "positive_prediction_count": positive_prediction_count,
        "negative_prediction_count": negative_prediction_count,
        "predicted_positive_rate": predicted_positive_rate,
        "predicted_negative_rate": predicted_negative_rate,
        "target_table": target_table,
        "target_partition_column": partition_column,
        "publish_enabled": publish_enabled,
        "dry_run": dry_run,
        "row_count_before_write": before_write_count,
        "row_count_after_write": after_write_count,
        "prewrite_validation": prewrite_summary,
        "postwrite_validation": postwrite_summary,
        "score_distribution": {row["metric"]: float(row["value"]) for _, row in score_distribution.iterrows()},
        "bucket_distribution": bucket_summary.to_dict("records"),
    }

    print(
        f"[SCORING] complete scored_rows={format_int(len(output_frame))} "
        f"publish_enabled={publish_enabled} dry_run={dry_run} "
        f"target={target_table} partition={partition_column}='{scoring_pt}'",
        flush=True,
    )


if __name__ == "__main__":
    main()
