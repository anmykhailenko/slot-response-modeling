from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.odps_reader import create_odps_client_from_env, load_odps_env_config, normalize_odps_table_name  # noqa: E402
from modeling.production_model_registry import resolve_production_model_reference  # noqa: E402


EXPECTED_TRAINING_TABLE = "pai_rec_prod.alg_uplift_phase1_response_dataset_di"
EXPECTED_SCORING_TABLE = "pai_rec_prod.alg_uplift_phase1_response_scores_di"
EXPECTED_MONITORING_DAILY_TABLE = "pai_rec_prod.ads_uplift_phase1_response_monitoring_daily_di"
EXPECTED_MONITORING_PERFORMANCE_TABLE = "pai_rec_prod.ads_uplift_phase1_response_monitoring_performance_di"
EXPECTED_MONITORING_ALERTS_TABLE = "pai_rec_prod.ads_uplift_phase1_response_monitoring_alerts_di"
FORBIDDEN_TABLE_REFERENCE_PATTERNS = (
    "response_predictions_di",
)


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_pt_not_future(name: str, value: str | None) -> None:
    if not value:
        return
    parsed = datetime.strptime(str(value), "%Y%m%d").date()
    if parsed > date.today():
        raise ValueError(f"{name}={value} is in the future relative to the runner date {date.today().isoformat()}.")


def validate_table_exists(table_name: str) -> None:
    project, table = normalize_odps_table_name(table_name)
    client = create_odps_client_from_env()
    if not client.exist_table(table, project=project):
        raise ValueError(f"Required ODPS table does not exist or is not visible: {project}.{table}")


def validate_tables_exist(table_names: Iterable[str]) -> None:
    for table_name in table_names:
        validate_table_exists(table_name)


def validate_config_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        check(path.exists(), f"Required config file is missing: {path}")
        check(path.is_file(), f"Expected a config file but found a non-file path: {path}")


def validate_no_forbidden_table_reference(*configs: Dict[str, Any]) -> None:
    serialized = "\n".join(yaml.safe_dump(config, sort_keys=True) for config in configs)
    for forbidden in FORBIDDEN_TABLE_REFERENCE_PATTERNS:
        if forbidden in serialized:
            raise ValueError(f"Forbidden retired prediction table reference detected in configs: {forbidden}")


def validate_training_config(config: Dict[str, Any]) -> Tuple[str, str]:
    training_table = str(config["source_table"])
    champion_reference_path = str(config["champion_reference_path"])
    check(
        training_table == EXPECTED_TRAINING_TABLE,
        f"Training source_table must be {EXPECTED_TRAINING_TABLE}, found {training_table}",
    )
    check(
        champion_reference_path == "contracts/model_registry/response_current.json",
        "Training champion_reference_path must be package-relative and point to contracts/model_registry/response_current.json",
    )
    check(
        str(config["target_column"]) == "response_label_positive_3d",
        "Training target_column must remain response_label_positive_3d",
    )
    split_policy = dict(config.get("split_policy", {}))
    check(bool(split_policy.get("require_contiguous_partitions", True)), "Time-based split must require contiguous partitions.")
    threshold_selection = dict(config.get("threshold_selection", {}))
    check(
        str(threshold_selection.get("policy")) == "max_precision_with_min_recall",
        "Threshold policy must remain max_precision_with_min_recall",
    )
    check(
        float(threshold_selection.get("minimum_recall", 0.0)) > 0.0,
        "Threshold policy must enforce a positive minimum recall constraint.",
    )
    if config.get("as_of_date") not in {None, ""}:
        validate_pt_not_future("training.as_of_date", str(config["as_of_date"]))
    return training_table, champion_reference_path


def validate_scoring_config(config: Dict[str, Any]) -> str:
    publish = dict(config["publish"])
    target_table = str(publish["target_table"])
    check(target_table == EXPECTED_SCORING_TABLE, f"Scoring target_table must be {EXPECTED_SCORING_TABLE}, found {target_table}")
    check(str(publish["partition_column"]) == "pt", "Scoring publish partition column must remain pt.")
    check(str(publish["write_mode"]).lower() == "overwrite", "Scoring publish write_mode must remain overwrite.")
    check(
        str(config["champion_reference_path"]) == "contracts/model_registry/response_current.json",
        "Scoring champion_reference_path must point to contracts/model_registry/response_current.json",
    )
    return target_table


def validate_monitoring_config(config: Dict[str, Any]) -> List[str]:
    monitoring = dict(config["response_monitoring"])
    source = dict(monitoring["source"])
    outputs = dict(monitoring["outputs"])
    check(
        str(source["scored_table"]) == EXPECTED_SCORING_TABLE,
        f"Monitoring scored_table must be {EXPECTED_SCORING_TABLE}, found {source['scored_table']}",
    )
    check(
        str(outputs["odps_daily_table"]) == EXPECTED_MONITORING_DAILY_TABLE,
        f"Monitoring daily output must be {EXPECTED_MONITORING_DAILY_TABLE}, found {outputs['odps_daily_table']}",
    )
    check(
        str(outputs["odps_performance_table"]) == EXPECTED_MONITORING_PERFORMANCE_TABLE,
        f"Monitoring performance output must be {EXPECTED_MONITORING_PERFORMANCE_TABLE}, found {outputs['odps_performance_table']}",
    )
    check(
        str(outputs["odps_alerts_table"]) == EXPECTED_MONITORING_ALERTS_TABLE,
        f"Monitoring alerts output must be {EXPECTED_MONITORING_ALERTS_TABLE}, found {outputs['odps_alerts_table']}",
    )
    return [
        str(source["scored_table"]),
        str(source["feature_table"]),
        str(source["outcome_source_table"]),
        str(outputs["odps_daily_table"]),
        str(outputs["odps_performance_table"]),
        str(outputs["odps_alerts_table"]),
    ]


def validate_model_reference(training_config: Dict[str, Any], require_model: bool) -> str:
    champion_reference_path = (PROJECT_ROOT / str(training_config["champion_reference_path"])).resolve()
    if not champion_reference_path.exists():
        if require_model:
            raise FileNotFoundError(
                "Champion reference does not exist yet. Run training first or place an approved model registry reference at "
                f"{champion_reference_path}"
            )
        return f"missing_optional:{champion_reference_path}"
    reference = resolve_production_model_reference(champion_reference_path=champion_reference_path)
    return f"resolved:{reference['model_name']}:{reference['model_version']}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run production preflight checks for the Slot Response Modeling handoff package.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "response_model.yaml")
    parser.add_argument("--pt", help="Optional scoring/monitoring partition to validate in YYYYMMDD format.")
    parser.add_argument("--reference-pt", dest="reference_pt", help="Optional explicit monitoring reference partition in YYYYMMDD format.")
    parser.add_argument("--require-model", action="store_true", help="Fail if the champion reference cannot be resolved.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    training_config_path = args.config.resolve()
    scoring_config_path = (PROJECT_ROOT / "configs" / "response_scoring.yaml").resolve()
    monitoring_config_path = (PROJECT_ROOT / "configs" / "response_monitoring.yaml").resolve()

    validate_config_paths([training_config_path, scoring_config_path, monitoring_config_path])
    validate_pt_not_future("pt", args.pt)
    validate_pt_not_future("reference_pt", args.reference_pt)
    if args.pt and args.reference_pt:
        check(args.reference_pt < args.pt, f"reference_pt must be earlier than pt: reference_pt={args.reference_pt} pt={args.pt}")

    training_config = load_yaml(training_config_path)
    scoring_config = load_yaml(scoring_config_path)
    monitoring_config = load_yaml(monitoring_config_path)
    validate_no_forbidden_table_reference(training_config, scoring_config, monitoring_config)

    load_odps_env_config()
    training_table, champion_reference_path = validate_training_config(training_config)
    scoring_table = validate_scoring_config(scoring_config)
    monitoring_tables = validate_monitoring_config(monitoring_config)
    validate_tables_exist([training_table, scoring_table, *monitoring_tables])
    model_status = validate_model_reference(training_config, require_model=args.require_model)

    print(f"[PREFLIGHT] odps_credentials=ok")
    print(f"[PREFLIGHT] configs=ok training={training_config_path} scoring={scoring_config_path} monitoring={monitoring_config_path}")
    print(f"[PREFLIGHT] training_table={training_table}")
    print(f"[PREFLIGHT] scoring_table={scoring_table}")
    print(f"[PREFLIGHT] monitoring_daily_table={EXPECTED_MONITORING_DAILY_TABLE}")
    print(f"[PREFLIGHT] monitoring_performance_table={EXPECTED_MONITORING_PERFORMANCE_TABLE}")
    print(f"[PREFLIGHT] monitoring_alerts_table={EXPECTED_MONITORING_ALERTS_TABLE}")
    print(f"[PREFLIGHT] champion_reference_path={champion_reference_path}")
    print(f"[PREFLIGHT] model_reference={model_status}")
    if args.pt:
        print(f"[PREFLIGHT] pt={args.pt}")
    if args.reference_pt:
        print(f"[PREFLIGHT] reference_pt={args.reference_pt}")
    print("[PREFLIGHT] status=ok")


if __name__ == "__main__":
    main()
