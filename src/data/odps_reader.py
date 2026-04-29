from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import pyarrow as pa  # type: ignore
except Exception:  # pragma: no cover
    pa = None  # type: ignore

try:
    from odps import ODPS  # type: ignore
except Exception:  # pragma: no cover
    ODPS = None  # type: ignore


LOGGER = logging.getLogger(__name__)
REQUIRED_ODPS_ENV_VARS = (
    "ALIBABA_CLOUD_ACCESS_KEY_ID",
    "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
    "ODPS_PROJECT",
    "ODPS_ENDPOINT",
)
DEPRECATED_ODPS_ENV_MAP = {
    "ALIBABA_CLOUD_ACCESS_KEY_ID": "ODPS_ACCESS_ID",
    "ALIBABA_CLOUD_ACCESS_KEY_SECRET": "ODPS_SECRET_ACCESS_KEY",
}


def _normalize_nullable_string(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_env_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    deprecated_name = DEPRECATED_ODPS_ENV_MAP.get(name)
    if not deprecated_name:
        return ""
    deprecated_value = os.getenv(deprecated_name, "").strip()
    if deprecated_value:
        LOGGER.warning(
            "Deprecated ODPS environment variable `%s` is in use. Set `%s` instead.",
            deprecated_name,
            name,
        )
    return deprecated_value


def load_odps_env_config() -> Dict[str, str]:
    config = {
        "access_id": _resolve_env_value("ALIBABA_CLOUD_ACCESS_KEY_ID"),
        "access_key": _resolve_env_value("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
        "project": os.getenv("ODPS_PROJECT", "").strip(),
        "endpoint": os.getenv("ODPS_ENDPOINT", "").strip(),
    }
    missing = [
        name
        for name, value in (
            ("ALIBABA_CLOUD_ACCESS_KEY_ID", config["access_id"]),
            ("ALIBABA_CLOUD_ACCESS_KEY_SECRET", config["access_key"]),
            ("ODPS_PROJECT", config["project"]),
            ("ODPS_ENDPOINT", config["endpoint"]),
        )
        if not value
    ]
    if missing:
        raise ValueError("Missing required ODPS environment variables: " + ", ".join(missing))
    return config


def create_odps_client_from_env() -> Any:
    if ODPS is None:
        raise ImportError("The `odps` package is required for uplift ODPS reads.")
    runtime = load_odps_env_config()
    return ODPS(
        access_id=runtime["access_id"],
        secret_access_key=runtime["access_key"],
        project=runtime["project"],
        endpoint=runtime["endpoint"],
    )


def normalize_odps_table_name(table_name: str, default_project: Optional[str] = None) -> Tuple[str, str]:
    normalized = _normalize_nullable_string(table_name)
    if not normalized:
        raise ValueError("Expected a non-empty ODPS table name.")
    if "." in normalized:
        project, table = [part.strip() for part in normalized.split(".", 1)]
        if not project or not table:
            raise ValueError(f"Invalid ODPS table reference: {table_name!r}")
        return project, table
    project = _normalize_nullable_string(default_project) or load_odps_env_config()["project"]
    return project, normalized


def _sql_literal(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _build_filters(filters: Optional[Dict[str, object]]) -> List[str]:
    clauses: List[str] = []
    for column, value in (filters or {}).items():
        if value is None:
            continue
        clauses.append(f"{column} = {_sql_literal(value)}")
    return clauses


def build_select_sql(
    *,
    table_name: str,
    columns: Sequence[str],
    filters: Optional[Dict[str, object]] = None,
    where_clauses: Optional[Iterable[str]] = None,
) -> str:
    if not columns:
        raise ValueError("At least one ODPS column must be selected.")
    selected_columns = ", ".join(dict.fromkeys(str(column).strip() for column in columns if str(column).strip()))
    project, table = normalize_odps_table_name(table_name)
    predicates = _build_filters(filters)
    predicates.extend(str(clause).strip() for clause in (where_clauses or []) if str(clause).strip())
    sql = f"select {selected_columns} from {project}.{table}"
    if predicates:
        sql = f"{sql} where {' and '.join(predicates)}"
    return sql


def fetch_sql_as_frame(
    sql: str,
    *,
    odps_client: Optional[Any] = None,
    batch_size: int = 500000,
    use_tunnel: bool = True,
    use_arrow: bool = True,
    arrow_diagnostic_enabled: bool = False,
    fallback_row_threshold: int = 2_000_000,
    expected_rows: Optional[int] = None,
    execution_details: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    client = odps_client or create_odps_client_from_env()
    LOGGER.info("Executing ODPS SQL: %s", sql)
    instance = client.execute_sql(sql)
    frames: List[pd.DataFrame] = []
    details = execution_details if execution_details is not None else {}
    details.update(
        {
            "arrow_enabled": bool(use_arrow),
            "arrow_diagnostic_enabled": bool(arrow_diagnostic_enabled),
            "fallback_row_threshold": int(fallback_row_threshold),
            "expected_rows": None if expected_rows is None else int(expected_rows),
            "fallback_used": False,
        }
    )

    if not use_arrow:
        LOGGER.info("ODPS Arrow reader disabled for this query; using non-Arrow reader directly.")
        with instance.open_reader(tunnel=use_tunnel, arrow=False) as reader:
            frames = list(reader.iter_pandas(batch_size=batch_size))
    else:
        try:
            with instance.open_reader(tunnel=use_tunnel, arrow=True) as reader:
                frames = list(reader.iter_pandas(batch_size=batch_size))
        except Exception as exc:
            is_arrow_error = False
            if pa is not None:
                arrow_error_types: tuple[type[BaseException], ...] = tuple(
                    error_type
                    for error_type in (
                        getattr(pa, "ArrowInvalid", None),
                        getattr(pa, "ArrowTypeError", None),
                        getattr(pa, "ArrowException", None),
                    )
                    if error_type is not None
                )
                is_arrow_error = isinstance(exc, arrow_error_types)
            if not is_arrow_error:
                raise

            expected_rows_text = "unknown" if expected_rows is None else f"{int(expected_rows):,}"
            if expected_rows is not None and expected_rows > int(fallback_row_threshold):
                raise RuntimeError(
                    "Arrow ODPS reader failed and non-Arrow fallback is blocked because "
                    f"expected_rows={expected_rows_text} exceeds fallback_row_threshold={int(fallback_row_threshold):,}."
                ) from exc

            LOGGER.warning(
                "Arrow ODPS reader failed; using non-Arrow fallback. expected_rows=%s threshold=%s error=%s",
                expected_rows_text,
                f"{int(fallback_row_threshold):,}",
                exc,
            )
            if arrow_diagnostic_enabled:
                LOGGER.warning("Arrow diagnostic requested but no additional probe is configured; proceeding with fallback.")

            details["fallback_used"] = True
            with instance.open_reader(tunnel=use_tunnel, arrow=False) as reader:
                frames = list(reader.iter_pandas(batch_size=batch_size))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_odps_table(
    *,
    table_name: str,
    columns: Sequence[str],
    filters: Optional[Dict[str, object]] = None,
    where_clauses: Optional[Iterable[str]] = None,
    odps_client: Optional[Any] = None,
    batch_size: int = 500000,
    use_arrow: bool = True,
    arrow_diagnostic_enabled: bool = False,
    fallback_row_threshold: int = 2_000_000,
    expected_rows: Optional[int] = None,
    execution_details: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    sql = build_select_sql(
        table_name=table_name,
        columns=columns,
        filters=filters,
        where_clauses=where_clauses,
    )
    return fetch_sql_as_frame(
        sql,
        odps_client=odps_client,
        batch_size=batch_size,
        use_arrow=use_arrow,
        arrow_diagnostic_enabled=arrow_diagnostic_enabled,
        fallback_row_threshold=fallback_row_threshold,
        expected_rows=expected_rows,
        execution_details=execution_details,
    )


def ensure_columns_exist(df: pd.DataFrame, required_columns: Sequence[str], context: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def save_local_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
