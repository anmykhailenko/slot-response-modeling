from __future__ import annotations

import argparse
import json
import logging
import math
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
    analysis_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(analysis_root))
    from profile_uplift_dataset import (
        _parse_dates,
        _resolve_step2_output_dir,
        load_step1_dataset,
        load_yaml_file,
        numeric_feature_columns,
        outcome_flag_columns,
        setup_logging,
    )
else:  # pragma: no cover
    from .profile_uplift_dataset import (
        _parse_dates,
        _resolve_step2_output_dir,
        load_step1_dataset,
        load_yaml_file,
        numeric_feature_columns,
        outcome_flag_columns,
        setup_logging,
    )


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _current_utc_date() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _date_from_optional(value: Optional[str]) -> pd.Timestamp:
    if value:
        return pd.Timestamp(value).normalize()
    return pd.Timestamp(_current_utc_date()).normalize()


def _treatment_combo_label(source_value: object, treatment_flag: object) -> str:
    if pd.isna(source_value):
        try:
            if int(treatment_flag) == 0:
                return "no_treatment"
        except Exception:
            pass
        return "unknown"
    token_set = {token.strip() for token in str(source_value).split("|") if token.strip()}
    if token_set == {"voucher"}:
        return "voucher_only"
    if token_set == {"sms"}:
        return "sms_only"
    if token_set == {"voucher", "sms"}:
        return "voucher_and_sms"
    if not token_set:
        return "no_treatment"
    return "other_multi_treatment"


def build_multi_treatment_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    working = dataset.copy()
    working["treatment_combo"] = [
        _treatment_combo_label(source_value, treatment_flag)
        for source_value, treatment_flag in zip(working.get("source_treatment_types"), working.get("treatment_flag"))
    ]
    summary = working["treatment_combo"].value_counts(dropna=False).rename_axis("treatment_combo").reset_index(name="row_count")
    summary["row_share"] = summary["row_count"] / max(len(working), 1)
    summary["unique_users"] = summary["treatment_combo"].map(
        working.groupby("treatment_combo")["player_id"].nunique(dropna=True).to_dict()
    )
    return summary.sort_values("treatment_combo").reset_index(drop=True)


def build_treatment_trend_tables(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    working = dataset.copy()
    working["assignment_date"] = _parse_dates(working["assignment_date"])
    working["assignment_week"] = working["assignment_date"] - pd.to_timedelta(working["assignment_date"].dt.weekday, unit="D")
    working["treatment_combo"] = [
        _treatment_combo_label(source_value, treatment_flag)
        for source_value, treatment_flag in zip(working.get("source_treatment_types"), working.get("treatment_flag"))
    ]

    daily = (
        working.groupby(["assignment_date", "treatment_combo"], dropna=False)
        .agg(row_count=("player_id", "size"), unique_users=("player_id", "nunique"))
        .reset_index()
        .sort_values(["assignment_date", "treatment_combo"])
    )
    weekly = (
        working.groupby(["assignment_week", "treatment_combo"], dropna=False)
        .agg(row_count=("player_id", "size"), unique_users=("player_id", "nunique"))
        .reset_index()
        .sort_values(["assignment_week", "treatment_combo"])
    )
    daily["assignment_date"] = daily["assignment_date"].dt.strftime("%Y-%m-%d")
    weekly["assignment_week"] = weekly["assignment_week"].dt.strftime("%Y-%m-%d")
    return daily, weekly


def build_outcome_maturity_summary(
    dataset: pd.DataFrame,
    *,
    base_config: Dict[str, Any],
    as_of_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assignment_date = _parse_dates(dataset["assignment_date"])
    rows: List[Dict[str, Any]] = []
    flags = pd.DataFrame({"player_id": dataset["player_id"], "assignment_date": assignment_date.dt.strftime("%Y-%m-%d")})
    for window in base_config.get("outcome_source", {}).get("outcome_windows", []):
        days = int(window["days"])
        column_prefix = f"outcome_{window['name']}_{days}d"
        matured_mask = assignment_date + pd.to_timedelta(days, unit="D") <= as_of_date
        incomplete_count = int((~matured_mask).sum())
        rows.append(
            {
                "outcome_window": f"{window['name']}_{days}d",
                "window_days": days,
                "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                "matured_rows": int(matured_mask.sum()),
                "incomplete_rows": incomplete_count,
                "matured_share": float(matured_mask.mean()) if len(dataset) else 0.0,
            }
        )
        flags[f"{column_prefix}_matured_flag"] = matured_mask.astype(int)
    return pd.DataFrame(rows), flags


def build_timing_violation_summary(
    dataset: pd.DataFrame,
    *,
    base_config: Dict[str, Any],
    contract: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assignment_date = _parse_dates(dataset["assignment_date"])
    feature_snapshot_date = _parse_dates(dataset["feature_snapshot_date"])
    first_outcome_event_date = _parse_dates(dataset["first_outcome_event_date"]) if "first_outcome_event_date" in dataset.columns else pd.Series(pd.NaT, index=dataset.index)
    last_outcome_event_date = _parse_dates(dataset["last_outcome_event_date"]) if "last_outcome_event_date" in dataset.columns else pd.Series(pd.NaT, index=dataset.index)
    minimum_feature_lag_days = int(contract.get("leakage_rules", {}).get("minimum_feature_lag_days", 1))

    feature_violation_mask = feature_snapshot_date >= assignment_date
    feature_lag_violation_mask = feature_snapshot_date > assignment_date - pd.to_timedelta(minimum_feature_lag_days, unit="D")
    outcome_start_violation_mask = first_outcome_event_date.notna() & (first_outcome_event_date <= assignment_date)
    outcome_end_violation_mask = last_outcome_event_date.notna() & (last_outcome_event_date <= assignment_date)

    summary = pd.DataFrame(
        [
            {"check": "feature_snapshot_not_strictly_before_assignment", "violation_rows": int(feature_violation_mask.sum())},
            {"check": "feature_snapshot_inside_minimum_lag_window", "violation_rows": int(feature_lag_violation_mask.sum())},
            {"check": "first_outcome_event_not_after_assignment", "violation_rows": int(outcome_start_violation_mask.sum())},
            {"check": "last_outcome_event_not_after_assignment", "violation_rows": int(outcome_end_violation_mask.sum())},
        ]
    )

    detail = pd.DataFrame(
        {
            "player_id": dataset["player_id"],
            "assignment_date": assignment_date.dt.strftime("%Y-%m-%d"),
            "feature_snapshot_date": feature_snapshot_date.dt.strftime("%Y-%m-%d"),
            "first_outcome_event_date": first_outcome_event_date.dt.strftime("%Y-%m-%d"),
            "last_outcome_event_date": last_outcome_event_date.dt.strftime("%Y-%m-%d"),
            "feature_snapshot_not_strictly_before_assignment": feature_violation_mask.astype(int),
            "feature_snapshot_inside_minimum_lag_window": feature_lag_violation_mask.astype(int),
            "first_outcome_event_not_after_assignment": outcome_start_violation_mask.astype(int),
            "last_outcome_event_not_after_assignment": outcome_end_violation_mask.astype(int),
        }
    )
    detail = detail.loc[
        (
            feature_violation_mask
            | feature_lag_violation_mask
            | outcome_start_violation_mask
            | outcome_end_violation_mask
        )
    ].reset_index(drop=True)
    return summary, detail


def _pooled_std(a: pd.Series, b: pd.Series) -> Optional[float]:
    a_clean = pd.to_numeric(a, errors="coerce").dropna()
    b_clean = pd.to_numeric(b, errors="coerce").dropna()
    if len(a_clean) < 2 or len(b_clean) < 2:
        return None
    a_var = float(a_clean.var(ddof=1))
    b_var = float(b_clean.var(ddof=1))
    pooled = math.sqrt((a_var + b_var) / 2.0)
    if pooled == 0:
        return None
    return pooled


def select_reference_group(dataset: pd.DataFrame) -> str:
    group_counts = dataset["treatment_group"].fillna("<null>").astype(str).value_counts()
    if "0" in group_counts.index:
        return "0"
    if "control" in group_counts.index:
        return "control"
    return str(group_counts.index[0])


def build_group_comparability_table(dataset: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    if not feature_columns:
        return pd.DataFrame(columns=["feature_name", "comparison_group", "reference_group", "reference_mean", "comparison_mean", "standardized_mean_difference", "missing_rate_gap", "imbalance_flag"])

    working = dataset.copy()
    working["treatment_group"] = working["treatment_group"].fillna("<null>").astype(str)
    reference_group = select_reference_group(working)
    reference = working.loc[working["treatment_group"] == reference_group].copy()
    if reference.empty:
        raise ValueError("Failed to identify a reference treatment group for comparability audit.")

    rows: List[Dict[str, Any]] = []
    for group_name in sorted(working["treatment_group"].unique().tolist()):
        if group_name == reference_group:
            continue
        comparison = working.loc[working["treatment_group"] == group_name].copy()
        for column in feature_columns:
            reference_numeric = pd.to_numeric(reference[column], errors="coerce")
            comparison_numeric = pd.to_numeric(comparison[column], errors="coerce")
            pooled = _pooled_std(reference_numeric, comparison_numeric)
            smd = None
            if pooled not in {None, 0}:
                smd = float((comparison_numeric.mean() - reference_numeric.mean()) / pooled)
            missing_gap = float(comparison_numeric.isna().mean() - reference_numeric.isna().mean())
            imbalance_flag = bool((smd is not None and abs(smd) >= 0.25) or abs(missing_gap) >= 0.10)
            rows.append(
                {
                    "feature_name": column,
                    "comparison_group": group_name,
                    "reference_group": reference_group,
                    "reference_mean": None if reference_numeric.dropna().empty else float(reference_numeric.mean()),
                    "comparison_mean": None if comparison_numeric.dropna().empty else float(comparison_numeric.mean()),
                    "reference_std": None if reference_numeric.dropna().empty else float(reference_numeric.std(ddof=1)),
                    "comparison_std": None if comparison_numeric.dropna().empty else float(comparison_numeric.std(ddof=1)),
                    "reference_missing_rate": float(reference_numeric.isna().mean()),
                    "comparison_missing_rate": float(comparison_numeric.isna().mean()),
                    "missing_rate_gap": missing_gap,
                    "standardized_mean_difference": smd,
                    "imbalance_flag": int(imbalance_flag),
                }
            )
    return pd.DataFrame(rows).sort_values(["imbalance_flag", "feature_name", "comparison_group"], ascending=[False, True, True]).reset_index(drop=True)


def build_recommendation(
    *,
    dataset: pd.DataFrame,
    maturity_summary: pd.DataFrame,
    timing_summary: pd.DataFrame,
    comparability_table: pd.DataFrame,
) -> Dict[str, Any]:
    combo_counts = build_multi_treatment_summary(dataset)
    combo_lookup = {row["treatment_combo"]: int(row["row_count"]) for _, row in combo_counts.iterrows()}
    has_control = combo_lookup.get("no_treatment", 0) > 0
    has_binary_treated = int(pd.to_numeric(dataset["treatment_flag"], errors="coerce").fillna(0).sum()) > 0
    has_multiple_non_control_groups = int(dataset["treatment_group"].fillna("<null>").astype(str).nunique()) >= 2
    maturity_ok = bool((maturity_summary["matured_share"] >= 0.95).all())
    timing_ok = bool((timing_summary["violation_rows"] == 0).all())
    imbalance_rate = float(comparability_table["imbalance_flag"].mean()) if not comparability_table.empty else 0.0
    imbalance_ok = imbalance_rate <= 0.20
    multi_combo_ok = combo_lookup.get("voucher_and_sms", 0) > 0 and combo_lookup.get("voucher_only", 0) > 0 and combo_lookup.get("sms_only", 0) > 0

    true_uplift_feasible = has_control and has_binary_treated and has_multiple_non_control_groups and maturity_ok and timing_ok and imbalance_ok
    binary_treatment_feasible = has_control and has_binary_treated and maturity_ok and timing_ok and imbalance_ok
    multi_treatment_feasible = has_control and multi_combo_ok and maturity_ok and timing_ok and imbalance_ok
    policy_evaluation_only = not binary_treatment_feasible

    reasons = []
    if not has_control:
        reasons.append("No clear untreated control rows are present in the Step 1 output.")
    if not maturity_ok:
        reasons.append("A material share of rows does not yet have fully matured outcome windows.")
    if not timing_ok:
        reasons.append("Timing audit found feature or outcome window violations that weaken causal interpretation.")
    if not imbalance_ok:
        reasons.append("Treatment groups are materially imbalanced on major pre-treatment features.")
    if not multi_combo_ok:
        reasons.append("Voucher-only, SMS-only, and combined-treatment cells are not all sufficiently represented for multi-treatment uplift.")
    if not reasons:
        reasons.append("The dataset currently shows control availability, mature outcomes, clean timing, and manageable pre-treatment imbalance.")

    return {
        "true_uplift_feasible": bool(true_uplift_feasible),
        "binary_treatment_uplift_feasible": bool(binary_treatment_feasible),
        "multi_treatment_uplift_feasible": bool(multi_treatment_feasible),
        "policy_evaluation_only": bool(policy_evaluation_only),
        "has_control_group": bool(has_control),
        "maturity_ok": bool(maturity_ok),
        "timing_ok": bool(timing_ok),
        "imbalance_ok": bool(imbalance_ok),
        "imbalance_flag_rate": imbalance_rate,
        "reasons": reasons,
    }


def write_step2_outputs(
    *,
    output_dir: Path,
    dataset: pd.DataFrame,
    multi_treatment_summary: pd.DataFrame,
    daily_trends: pd.DataFrame,
    weekly_trends: pd.DataFrame,
    maturity_summary: pd.DataFrame,
    maturity_flags: pd.DataFrame,
    timing_summary: pd.DataFrame,
    timing_detail: pd.DataFrame,
    comparability_table: pd.DataFrame,
    recommendation: Dict[str, Any],
    partition_value: Optional[str],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "multi_treatment_summary": output_dir / "multi_treatment_summary.csv",
        "daily_treatment_trends": output_dir / "daily_treatment_trends.csv",
        "weekly_treatment_trends": output_dir / "weekly_treatment_trends.csv",
        "outcome_maturity_summary": output_dir / "outcome_maturity_summary.csv",
        "outcome_maturity_flags": output_dir / "outcome_maturity_flags.csv",
        "timing_violation_summary": output_dir / "timing_violation_summary.csv",
        "timing_violation_detail": output_dir / "timing_violation_detail.csv",
        "group_comparability": output_dir / "group_comparability.csv",
        "experiment_design_audit": output_dir / "experiment_design_audit.json",
        "experiment_design_audit_md": output_dir / "experiment_design_audit.md",
    }

    multi_treatment_summary.to_csv(artifacts["multi_treatment_summary"], index=False)
    daily_trends.to_csv(artifacts["daily_treatment_trends"], index=False)
    weekly_trends.to_csv(artifacts["weekly_treatment_trends"], index=False)
    maturity_summary.to_csv(artifacts["outcome_maturity_summary"], index=False)
    maturity_flags.to_csv(artifacts["outcome_maturity_flags"], index=False)
    timing_summary.to_csv(artifacts["timing_violation_summary"], index=False)
    timing_detail.to_csv(artifacts["timing_violation_detail"], index=False)
    comparability_table.to_csv(artifacts["group_comparability"], index=False)

    json_payload = {
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "partition_value": partition_value,
        "row_count": int(len(dataset)),
        "recommendation": recommendation,
    }
    artifacts["experiment_design_audit"].write_text(json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    top_imbalances = comparability_table.loc[comparability_table["imbalance_flag"] == 1].head(10)
    lines = [
        "# Step 2 Experiment Design Audit",
        "",
        f"- Partition: `{partition_value or 'all_available'}`",
        f"- Rows audited: {len(dataset)}",
        f"- True uplift feasible: `{str(recommendation['true_uplift_feasible']).lower()}`",
        f"- Binary treatment uplift feasible: `{str(recommendation['binary_treatment_uplift_feasible']).lower()}`",
        f"- Multi-treatment uplift feasible: `{str(recommendation['multi_treatment_uplift_feasible']).lower()}`",
        f"- Policy evaluation only: `{str(recommendation['policy_evaluation_only']).lower()}`",
        "",
        "## Why",
    ]
    lines.extend([f"- {reason}" for reason in recommendation["reasons"]])
    lines.extend(["", "## Largest Imbalances"])
    if top_imbalances.empty:
        lines.append("- No major imbalances exceeded the configured practical thresholds.")
    else:
        for _, row in top_imbalances.iterrows():
            smd = row["standardized_mean_difference"]
            lines.append(
                f"- `{row['feature_name']}` {row['reference_group']} vs {row['comparison_group']}: "
                f"SMD={smd:.3f} missing_gap={row['missing_rate_gap']:.3f}"
            )
    artifacts["experiment_design_audit_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifacts


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Audit whether the Step 1 uplift dataset supports uplift or only policy evaluation.")
    parser.add_argument("--config-path", type=Path, default=project_root / "configs/base.yaml")
    parser.add_argument("--contract-path", type=Path, default=project_root / "configs/data_contract.yaml")
    parser.add_argument("--partition-value", help="Optional Step 1 output partition to audit.")
    parser.add_argument("--as-of-date", help="Outcome maturity cutoff date in YYYY-MM-DD. Defaults to current UTC date.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory override.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    logger = setup_logging(args.log_level)
    base_config = load_yaml_file(args.config_path)
    contract = load_yaml_file(args.contract_path)
    dataset = load_step1_dataset(base_config=base_config, contract=contract, partition_value=args.partition_value)
    as_of_date = _date_from_optional(args.as_of_date)

    multi_treatment_summary = build_multi_treatment_summary(dataset)
    daily_trends, weekly_trends = build_treatment_trend_tables(dataset)
    maturity_summary, maturity_flags = build_outcome_maturity_summary(dataset, base_config=base_config, as_of_date=as_of_date)
    timing_summary, timing_detail = build_timing_violation_summary(dataset, base_config=base_config, contract=contract)
    feature_columns = numeric_feature_columns(base_config, dataset)
    comparability_table = build_group_comparability_table(dataset, feature_columns)
    recommendation = build_recommendation(
        dataset=dataset,
        maturity_summary=maturity_summary,
        timing_summary=timing_summary,
        comparability_table=comparability_table,
    )

    output_dir = args.output_dir or _resolve_step2_output_dir(base_config)
    artifacts = write_step2_outputs(
        output_dir=output_dir,
        dataset=dataset,
        multi_treatment_summary=multi_treatment_summary,
        daily_trends=daily_trends,
        weekly_trends=weekly_trends,
        maturity_summary=maturity_summary,
        maturity_flags=maturity_flags,
        timing_summary=timing_summary,
        timing_detail=timing_detail,
        comparability_table=comparability_table,
        recommendation=recommendation,
        partition_value=args.partition_value,
    )
    logger.info(
        "Experiment audit completed rows=%s true_uplift=%s binary=%s multi=%s output_dir=%s",
        len(dataset),
        recommendation["true_uplift_feasible"],
        recommendation["binary_treatment_uplift_feasible"],
        recommendation["multi_treatment_uplift_feasible"],
        output_dir,
    )
    return {
        "row_count": int(len(dataset)),
        "recommendation": recommendation,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }


if __name__ == "__main__":
    main()
