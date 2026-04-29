from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

if __package__ in {None, "", "monitoring"}:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from data.odps_reader import create_odps_client_from_env, fetch_sql_as_frame, normalize_odps_table_name
    from data.odps_writer import write_frame_to_odps
else:  # pragma: no cover
    from ..data.odps_reader import create_odps_client_from_env, fetch_sql_as_frame, normalize_odps_table_name
    from ..data.odps_writer import write_frame_to_odps


def fetch_partition_frame(
    *,
    table_name: str,
    partition_column: str,
    partition_value: str,
    columns: Iterable[str],
    extra_where_clauses: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    selected_columns = ", ".join(dict.fromkeys(str(column).strip() for column in columns if str(column).strip()))
    clauses = [f"{partition_column} = '{partition_value}'"]
    clauses.extend(str(clause).strip() for clause in (extra_where_clauses or []) if str(clause).strip())
    sql = f"select {selected_columns} from {table_name} where {' and '.join(clauses)}"
    return fetch_sql_as_frame(sql, odps_client=create_odps_client_from_env())


def fetch_scalar(sql: str, column_name: str) -> Any:
    frame = fetch_sql_as_frame(sql, odps_client=create_odps_client_from_env(), batch_size=10000)
    if frame.empty:
        return None
    return frame.iloc[0][column_name]


def list_partition_values(table_name: str) -> List[str]:
    project, table = normalize_odps_table_name(table_name)
    client = create_odps_client_from_env()
    odps_table = client.get_table(table, project=project)
    values: List[str] = []
    for partition in odps_table.partitions:
        spec = str(partition.partition_spec)
        start = spec.find("'")
        end = spec.rfind("'")
        if start == -1 or end == -1 or end <= start:
            continue
        values.append(spec[start + 1 : end])
    return sorted(values)


def fetch_table_column_names(table_name: str) -> List[str]:
    project, table = normalize_odps_table_name(table_name)
    client = create_odps_client_from_env()
    odps_table = client.get_table(table, project=project)
    return [column.name for column in odps_table.table_schema.simple_columns]


def build_partition_exists_sql(table_name: str, partition_column: str, partition_value: str) -> str:
    return (
        f"select count(1) as row_count from {table_name} "
        f"where {partition_column} = '{partition_value}'"
    )


def fetch_eligible_population_summary(
    *,
    table_name: str,
    partition_column: str,
    partition_value: str,
    eligibility_filter_sql: Optional[str],
    segment_column: str,
) -> pd.DataFrame:
    clauses = [f"{partition_column} = '{partition_value}'"]
    if eligibility_filter_sql:
        clauses.append(f"({eligibility_filter_sql})")
    sql = f"""
select
    cast(coalesce({segment_column}, '__missing__') as string) as segment_value,
    count(1) as eligible_row_count,
    count(distinct cast(player_id as string)) as eligible_distinct_player_count,
    min(cast(snapshot_date as date)) as snapshot_date_min,
    max(cast(snapshot_date as date)) as snapshot_date_max
from {table_name}
where {' and '.join(clauses)}
group by cast(coalesce({segment_column}, '__missing__') as string)
order by eligible_row_count desc, segment_value asc
"""
    return fetch_sql_as_frame(sql, odps_client=create_odps_client_from_env())


def fetch_observational_outcomes_for_scored_partition(
    *,
    scored_table: str,
    scored_partition_column: str,
    scoring_pt: str,
    outcome_source_table: str,
    outcome_partition_column: str,
    response_window_days: int,
) -> pd.DataFrame:
    score_date = datetime.strptime(scoring_pt, "%Y%m%d").date()
    min_outcome_pt = (score_date + timedelta(days=1)).strftime("%Y%m%d")
    max_outcome_pt = (score_date + timedelta(days=response_window_days)).strftime("%Y%m%d")
    sql = f"""
with scored as (
    select
        cast(player_id as string) as player_id,
        cast(score_date as date) as score_date,
        cast(predicted_response_score as double) as predicted_response_score,
        cast(response_priority_bucket as string) as response_priority_bucket,
        cast(vip_level as string) as vip_level,
        cast(selected_threshold as double) as selected_threshold,
        cast(model_name as string) as model_name,
        cast(model_version as string) as model_version,
        cast(scoring_pt as string) as scoring_pt
    from {scored_table}
    where {scored_partition_column} = '{scoring_pt}'
),
filtered_outcomes as (
    select
        login_name,
        bet_amount,
        stat_date,
        {outcome_partition_column}
    from {outcome_source_table}
    where {outcome_partition_column} >= '{min_outcome_pt}'
      and {outcome_partition_column} <= '{max_outcome_pt}'
),
outcomes as (
    select
        s.player_id,
        s.scoring_pt,
        sum(cast(o.bet_amount as double)) as observed_gross_bet_value_{response_window_days}d,
        count(o.login_name) as observed_outcome_source_rows_{response_window_days}d,
        min(cast(o.stat_date as date)) as first_observed_outcome_date,
        max(cast(o.stat_date as date)) as last_observed_outcome_date
    from scored s
    left join filtered_outcomes o
      on s.player_id = cast(o.login_name as string)
     and cast(o.stat_date as date) > s.score_date
     and cast(o.stat_date as date) <= dateadd(s.score_date, {response_window_days}, 'dd')
    group by
        s.player_id,
        s.scoring_pt
)
select
    s.player_id,
    s.scoring_pt as pt,
    s.predicted_response_score,
    s.response_priority_bucket,
    s.vip_level,
    s.selected_threshold,
    s.model_name,
    s.model_version,
    cast(coalesce(o.observed_gross_bet_value_{response_window_days}d, 0.0) as double) as observed_gross_bet_value_{response_window_days}d,
    case when coalesce(o.observed_gross_bet_value_{response_window_days}d, 0.0) > 0 then cast(1 as bigint) else cast(0 as bigint) end as observed_response_label_positive_{response_window_days}d,
    cast(coalesce(o.observed_outcome_source_rows_{response_window_days}d, 0) as bigint) as observed_outcome_source_rows_{response_window_days}d,
    o.first_observed_outcome_date,
    o.last_observed_outcome_date
from scored s
left join outcomes o
  on s.player_id = o.player_id
 and s.scoring_pt = o.scoring_pt
"""
    return fetch_sql_as_frame(sql, odps_client=create_odps_client_from_env())


def write_monitoring_frame_to_odps(
    df: pd.DataFrame,
    *,
    odps_target: str,
    partition_column: str,
    partition_value: str,
    write_mode: str = "overwrite",
) -> None:
    write_frame_to_odps(
        df,
        odps_target=odps_target,
        partition_column=partition_column,
        partition_value=partition_value,
        write_mode=write_mode,
    )


def serialize_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def alerts_to_frame(alerts: List[Dict[str, Any]], *, pt: str, mode: str, run_label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for alert in alerts:
        rows.append(
            {
                "monitor_run_ts": alert.get("monitor_run_ts"),
                "run_label": run_label,
                "mode": str(mode),
                "severity": None if alert.get("severity") is None else str(alert.get("severity")),
                "check_name": None if alert.get("check_name") is None else str(alert.get("check_name")),
                "metric_name": None if alert.get("metric_name") is None else str(alert.get("metric_name")),
                "observed_value": None if alert.get("observed_value") is None else str(alert.get("observed_value")),
                "threshold_value": None if alert.get("threshold_value") is None else str(alert.get("threshold_value")),
                "reference_value": None if alert.get("reference_value") is None else str(alert.get("reference_value")),
                "message": None if alert.get("message") is None else str(alert.get("message")),
                "context_json": serialize_json(alert.get("context", {})),
                "pt": str(pt),
            }
        )
    columns = [
        "monitor_run_ts",
        "run_label",
        "mode",
        "severity",
        "check_name",
        "metric_name",
        "observed_value",
        "threshold_value",
        "reference_value",
        "message",
        "context_json",
        "pt",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).loc[:, columns]


def ensure_table_reference(table_name: str) -> str:
    project, table = normalize_odps_table_name(table_name)
    return f"{project}.{table}"
