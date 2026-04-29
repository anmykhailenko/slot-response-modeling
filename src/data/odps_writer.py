from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

if __package__ in {None, ""}:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from data.odps_reader import create_odps_client_from_env
else:  # pragma: no cover
    from .odps_reader import create_odps_client_from_env


ALLOWED_WRITE_MODES = {"append", "overwrite"}


def _normalize_nullable_string(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_odps_target(
    *,
    odps_target: Optional[str] = None,
    odps_project: Optional[str] = None,
    odps_table: Optional[str] = None,
) -> Tuple[str, str]:
    project = _normalize_nullable_string(odps_project)
    table = _normalize_nullable_string(odps_table)
    target = _normalize_nullable_string(odps_target)

    if project and table:
        return project, table
    if target:
        parts = [part.strip() for part in target.split(".", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("ODPS target must be fully qualified as `project.table`.")
        if project is None:
            project = parts[0]
        if table is None:
            table = parts[1]
    if not project:
        raise ValueError("ODPS target requires a project.")
    if not table:
        raise ValueError("ODPS target requires a table.")
    return project, table


def validate_odps_write_config(
    *,
    odps_project: str,
    odps_table: str,
    partition_column: str,
    partition_value: str,
    write_mode: str,
) -> None:
    if not str(odps_project).strip():
        raise ValueError("ODPS output requires `odps_project`.")
    if not str(odps_table).strip():
        raise ValueError("ODPS output requires `odps_table`.")
    if not str(partition_column).strip():
        raise ValueError("ODPS output requires `partition_column`.")
    if not str(partition_value).strip():
        raise ValueError("ODPS output requires `partition_value`.")
    normalized_mode = str(write_mode).strip().lower()
    if normalized_mode not in ALLOWED_WRITE_MODES:
        raise ValueError(f"Unsupported ODPS write mode `{write_mode}`. Expected one of {sorted(ALLOWED_WRITE_MODES)}.")


def build_partition_spec(partition_column: str, partition_value: str) -> str:
    return f"{partition_column.strip()}={partition_value.strip()}"


def _parse_datetime_like(value: object) -> Optional[datetime]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _parse_date_like(value: object) -> Optional[date]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    return pd.to_datetime(text, errors="raise").date()


def _normalize_odps_scalar(value: object, odps_type_name: Optional[str] = None) -> object:
    if value is None:
        return None
    if pd.isna(value):
        return None
    normalized_type = (odps_type_name or "").strip().lower()
    if normalized_type == "timestamp":
        return _parse_datetime_like(value)
    if normalized_type == "datetime":
        return _parse_datetime_like(value)
    if normalized_type == "date":
        return _parse_date_like(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _iter_odps_rows(frame: pd.DataFrame, odps_type_names: list[str]):
    for row in frame.itertuples(index=False, name=None):
        yield tuple(
            _normalize_odps_scalar(value, odps_type_name=odps_type_names[idx] if idx < len(odps_type_names) else None)
            for idx, value in enumerate(row)
        )


def write_frame_to_odps(
    df: pd.DataFrame,
    *,
    odps_target: Optional[str] = None,
    odps_project: Optional[str] = None,
    odps_table: Optional[str] = None,
    partition_column: str,
    partition_value: str,
    write_mode: str = "overwrite",
) -> None:
    resolved_project, resolved_table = normalize_odps_target(
        odps_target=odps_target,
        odps_project=odps_project,
        odps_table=odps_table,
    )
    validate_odps_write_config(
        odps_project=resolved_project,
        odps_table=resolved_table,
        partition_column=partition_column,
        partition_value=partition_value,
        write_mode=write_mode,
    )
    client = create_odps_client_from_env()
    if not client.exist_table(resolved_table, project=resolved_project):
        raise ValueError(
            "Target ODPS table does not exist. "
            f"Project='{resolved_project}', Table='{resolved_table}', {partition_column}='{partition_value}'"
        )
    target_table = client.get_table(resolved_table, project=resolved_project)
    payload_columns = [column.name for column in target_table.table_schema.simple_columns]
    payload = df.drop(columns=[partition_column], errors="ignore").copy()
    payload = payload.reindex(columns=payload_columns)
    odps_type_names = [str(column.type).strip().lower() for column in target_table.table_schema.simple_columns]
    try:
        client.write_table(
            resolved_table,
            _iter_odps_rows(payload, odps_type_names),
            project=resolved_project,
            partition=build_partition_spec(partition_column, partition_value),
            create_partition=True,
            overwrite=str(write_mode).strip().lower() == "overwrite",
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to write uplift dataset to ODPS. "
            f"Project='{resolved_project}', Table='{resolved_table}', {partition_column}='{partition_value}'"
        ) from exc
