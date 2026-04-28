from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from data.odps_reader import read_odps_table
else:  # pragma: no cover
    from ..data.odps_reader import read_odps_table


DEFAULT_OUTPUT_SUBDIR = "step2"


def setup_logging(level_name: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("uplift.profile")


def load_yaml_file(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to load uplift YAML config files.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _normalize_nullable_string(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _current_utc_date() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _resolve_step2_output_dir(base_config: Dict[str, Any]) -> Path:
    root = Path(str(base_config.get("runtime", {}).get("local_output_dir", "response_modeling/outputs")))
    return root / DEFAULT_OUTPUT_SUBDIR


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def outcome_flag_columns(contract: Dict[str, Any]) -> List[str]:
    return [str(column) for column in contract.get("required_outcome_columns", [])]


def numeric_feature_columns(base_config: Dict[str, Any], dataset: pd.DataFrame) -> List[str]:
    feature_columns = [str(column) for column in base_config.get("feature_source", {}).get("select_columns", [])]
    numeric_columns: List[str] = []
    for column in feature_columns:
        if column in dataset.columns and pd.api.types.is_numeric_dtype(dataset[column]):
            numeric_columns.append(column)
    return numeric_columns


def load_step1_dataset(
    *,
    base_config: Dict[str, Any],
    contract: Dict[str, Any],
    partition_value: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    output_target = _normalize_nullable_string(base_config.get("output", {}).get("odps_target"))
    if not output_target:
        raise ValueError("output.odps_target must be configured in Step 1 base config.")
    selected_columns = list(columns) if columns else _recommended_profile_columns(base_config, contract)
    build_partition_column = str(base_config.get("dataset", {}).get("build_partition_column", "pt"))
    where_clauses = []
    if partition_value:
        where_clauses.append(f"{build_partition_column} = '{partition_value}'")
    dataset = read_odps_table(table_name=output_target, columns=selected_columns, where_clauses=where_clauses)
    if dataset.empty:
        raise ValueError("No rows were returned from the Step 1 ODPS uplift dataset.")
    return dataset


def _recommended_profile_columns(base_config: Dict[str, Any], contract: Dict[str, Any]) -> List[str]:
    columns = list(contract.get("required_columns", []))
    columns.extend(outcome_flag_columns(contract))
    columns.extend(str(column) for column in contract.get("audit_columns", []))
    columns.extend(str(column) for column in base_config.get("feature_source", {}).get("select_columns", []))
    columns.extend(["treatment_value", "dataset_build_ts", "pt"])
    return list(dict.fromkeys(columns))


def build_overview_table(dataset: pd.DataFrame) -> pd.DataFrame:
    assignment_dates = _parse_dates(dataset["assignment_date"])
    rows = [
        {"metric": "row_count", "value": int(len(dataset))},
        {"metric": "unique_users", "value": int(dataset["player_id"].nunique(dropna=True))},
        {
            "metric": "assignment_date_min",
            "value": None if assignment_dates.dropna().empty else assignment_dates.min().strftime("%Y-%m-%d"),
        },
        {
            "metric": "assignment_date_max",
            "value": None if assignment_dates.dropna().empty else assignment_dates.max().strftime("%Y-%m-%d"),
        },
        {"metric": "distinct_assignment_dates", "value": int(assignment_dates.nunique(dropna=True))},
    ]
    return pd.DataFrame(rows)


def build_distribution_table(dataset: pd.DataFrame, column_name: str) -> pd.DataFrame:
    counts = dataset[column_name].fillna("<null>").astype(str).value_counts(dropna=False).rename_axis(column_name).reset_index(name="row_count")
    counts["row_share"] = counts["row_count"] / max(len(dataset), 1)
    return counts


def build_outcome_distribution_table(dataset: pd.DataFrame, outcome_columns: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for column in outcome_columns:
        numeric = pd.to_numeric(dataset[column], errors="coerce")
        distribution = numeric.fillna(-999999).value_counts(dropna=False).to_dict()
        rows.append(
            {
                "outcome_column": column,
                "mean": float(numeric.mean()) if not numeric.dropna().empty else None,
                "missing_rate": float(numeric.isna().mean()),
                "distribution_json": json.dumps({str(key): int(value) for key, value in distribution.items()}, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def build_missingness_table(dataset: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        rows.append(
            {
                "column_name": column,
                "missing_count": int(dataset[column].isna().sum()) if column in dataset.columns else len(dataset),
                "missing_rate": float(dataset[column].isna().mean()) if column in dataset.columns else 1.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "column_name"], ascending=[False, True]).reset_index(drop=True)


def build_assignment_coverage_table(dataset: pd.DataFrame) -> pd.DataFrame:
    working = dataset.copy()
    working["assignment_date"] = _parse_dates(working["assignment_date"])
    coverage = (
        working.groupby("assignment_date", dropna=False)
        .agg(row_count=("player_id", "size"), unique_users=("player_id", "nunique"))
        .reset_index()
        .sort_values("assignment_date")
    )
    coverage["assignment_date"] = coverage["assignment_date"].dt.strftime("%Y-%m-%d")
    return coverage


def build_profile_outputs(dataset: pd.DataFrame, *, base_config: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    outcome_columns = outcome_flag_columns(contract)
    key_fields = list(dict.fromkeys(contract.get("required_columns", []) + outcome_columns + ["first_outcome_event_date", "last_outcome_event_date", "pt"]))
    return {
        "overview": build_overview_table(dataset),
        "assignment_coverage_daily": build_assignment_coverage_table(dataset),
        "treatment_flag_distribution": build_distribution_table(dataset, "treatment_flag"),
        "treatment_group_distribution": build_distribution_table(dataset, "treatment_group"),
        "treatment_type_distribution": build_distribution_table(dataset, "treatment_type"),
        "outcome_distribution": build_outcome_distribution_table(dataset, outcome_columns),
        "missingness_key_fields": build_missingness_table(dataset, key_fields),
    }


def write_profile_outputs(
    outputs: Dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    partition_value: Optional[str],
    dataset_name: str,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for name, frame in outputs.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path

    markdown_path = output_dir / "dataset_profile_summary.md"
    json_path = output_dir / "dataset_profile_summary.json"
    summary = {
        "dataset_name": dataset_name,
        "partition_value": partition_value,
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "artifacts": {name: str(path) for name, path in written.items()},
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Step 2 Dataset Profile",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Partition: `{partition_value or 'all_available'}`",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## CSV Outputs",
    ]
    for name, path in written.items():
        lines.append(f"- `{name}`: `{path.name}`")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written["markdown_summary"] = markdown_path
    written["json_summary"] = json_path
    return written


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Profile the Step 1 uplift dataset from ODPS.")
    parser.add_argument("--config-path", type=Path, default=project_root / "configs/base.yaml")
    parser.add_argument("--contract-path", type=Path, default=project_root / "configs/data_contract.yaml")
    parser.add_argument("--partition-value", help="Optional Step 1 output partition to profile.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory override.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    logger = setup_logging(args.log_level)
    base_config = load_yaml_file(args.config_path)
    contract = load_yaml_file(args.contract_path)
    dataset = load_step1_dataset(base_config=base_config, contract=contract, partition_value=args.partition_value)
    outputs = build_profile_outputs(dataset, base_config=base_config, contract=contract)
    output_dir = args.output_dir or _resolve_step2_output_dir(base_config)
    dataset_name = str(base_config.get("dataset", {}).get("dataset_name", "uplift_dataset"))
    artifacts = write_profile_outputs(outputs, output_dir=output_dir, partition_value=args.partition_value, dataset_name=dataset_name)
    logger.info("Profile completed rows=%s output_dir=%s", len(dataset), output_dir)
    return {"row_count": int(len(dataset)), "artifacts": {name: str(path) for name, path in artifacts.items()}}


if __name__ == "__main__":
    main()
