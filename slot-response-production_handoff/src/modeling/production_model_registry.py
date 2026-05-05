from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_CHAMPION_REFERENCE_PATH = Path("contracts/model_registry/response_current.json")
REQUIRED_MODEL_FILES = (
    "model.joblib",
    "preprocessor.joblib",
    "preprocessing.json",
    "threshold_selection.json",
)
REQUIRED_RUN_METADATA_FIELDS = (
    "model_name",
    "model_version",
    "run_id",
    "iteration_id",
    "selected_threshold",
    "selected_score_variant",
    "export_bundle_path",
)
REQUIRED_CHAMPION_FIELDS = (
    "model_dir",
    "export_bundle_path",
    "model_name",
    "model_version",
    "mlflow_run_id",
    "iteration_id",
    "threshold_artifact_path",
    "feature_schema_path",
    "run_metadata_path",
    "selected_threshold",
    "selected_score_variant",
)

LOCAL_RUN_ID_PREFIXES = ("local_", "localbundle", "local-bundle", "local_bundle")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_model_version(*, iteration_id: str, model_name: str, mlflow_run_id: str) -> str:
    iteration = str(iteration_id).strip()
    model = str(model_name).strip()
    run_id = str(mlflow_run_id).strip()
    if not iteration or not model or not run_id:
        raise ValueError("model_version requires non-empty iteration_id, model_name, and mlflow_run_id.")
    return f"{iteration}__{model}__{run_id[:8]}"


def _normalize_nullable_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _candidate_layouts(path: Path) -> Iterable[Dict[str, Path]]:
    resolved = path.expanduser().resolve()
    yield {"bundle_dir": resolved, "model_dir": resolved}
    yield {"bundle_dir": resolved, "model_dir": resolved / "model"}
    if resolved.name == "model":
        yield {"bundle_dir": resolved.parent, "model_dir": resolved}


def _validate_model_files(model_dir: Path) -> None:
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Model directory {model_dir} is missing required production inference artifacts: {missing}"
        )


def _resolve_feature_schema_path(bundle_dir: Path, model_dir: Path) -> Path:
    candidates = [model_dir / "feature_schema.json", bundle_dir / "feature_schema.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Bundle {bundle_dir} is missing `feature_schema.json` in both bundle root and model directory."
    )


def _validate_run_metadata_payload(payload: Dict[str, Any], run_metadata_path: Path) -> None:
    missing = [field for field in REQUIRED_RUN_METADATA_FIELDS if not _normalize_nullable_string(payload.get(field))]
    if missing:
        raise ValueError(f"Run metadata at {run_metadata_path} is missing required fields: {missing}")


def _is_local_bundle_metadata(payload: Dict[str, Any], bundle_dir: Path) -> bool:
    run_id = (_normalize_nullable_string(payload.get("run_id")) or "").lower()
    export_bundle_path = (_normalize_nullable_string(payload.get("export_bundle_path")) or "").lower()
    bundle_name = bundle_dir.name.lower()
    return (
        run_id.startswith(LOCAL_RUN_ID_PREFIXES)
        or bundle_name.endswith("_local")
        or export_bundle_path.endswith("_local")
    )


def build_reference_from_bundle(candidate_path: Path) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for layout in _candidate_layouts(candidate_path):
        bundle_dir = layout["bundle_dir"]
        model_dir = layout["model_dir"]
        try:
            if not bundle_dir.exists() or not bundle_dir.is_dir():
                raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")
            if not model_dir.exists() or not model_dir.is_dir():
                raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
            _validate_model_files(model_dir)
            run_metadata_path = bundle_dir / "run_metadata.json"
            if not run_metadata_path.exists():
                raise FileNotFoundError(f"Bundle directory {bundle_dir} is missing run_metadata.json")
            run_metadata = load_json(run_metadata_path)
            _validate_run_metadata_payload(run_metadata, run_metadata_path)
            if _is_local_bundle_metadata(run_metadata, bundle_dir):
                raise ValueError(
                    "Bundle is not production-safe because it is marked as a local/dev export. "
                    f"bundle_dir={bundle_dir} run_id={run_metadata.get('run_id')!r}"
                )
            feature_schema_path = _resolve_feature_schema_path(bundle_dir, model_dir)
            threshold_artifact_path = model_dir / "threshold_selection.json"
            preprocessing_path = model_dir / "preprocessing.json"
            return {
                "resolution_source": "bundle",
                "bundle_dir": str(bundle_dir),
                "model_dir": str(model_dir),
                "feature_schema_path": str(feature_schema_path),
                "threshold_artifact_path": str(threshold_artifact_path),
                "preprocessing_path": str(preprocessing_path),
                "run_metadata_path": str(run_metadata_path),
                "model_name": str(run_metadata["model_name"]),
                "model_version": str(run_metadata["model_version"]),
                "mlflow_run_id": str(run_metadata["run_id"]),
                "iteration_id": str(run_metadata["iteration_id"]),
                "selected_threshold": float(run_metadata["selected_threshold"]),
                "selected_score_variant": str(run_metadata["selected_score_variant"]),
                "export_bundle_path": str(bundle_dir),
            }
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"Unable to resolve bundle layout from {candidate_path}")


def write_champion_reference(path: Path, payload: Dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_CHAMPION_FIELDS if not _normalize_nullable_string(payload.get(field))]
    if missing:
        raise ValueError(f"Champion reference is missing required fields: {missing}")
    validated = {
        **payload,
        "updated_at": payload.get("updated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(path, validated)


def resolve_champion_reference(reference_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if reference_path is None:
        return None
    resolved_reference = reference_path.expanduser().resolve()
    if not resolved_reference.exists():
        return None
    payload = load_json(resolved_reference)
    missing = [field for field in REQUIRED_CHAMPION_FIELDS if not _normalize_nullable_string(payload.get(field))]
    if missing:
        raise ValueError(f"Champion reference at {resolved_reference} is missing required fields: {missing}")
    model_dir = Path(str(payload["model_dir"])).expanduser()
    if not model_dir.is_absolute():
        model_dir = (resolved_reference.parent / model_dir).resolve()
    bundle_reference = build_reference_from_bundle(model_dir)
    reference = {
        **payload,
        **bundle_reference,
        "resolution_source": "champion",
        "champion_reference_path": str(resolved_reference),
    }
    return reference


def resolve_production_model_reference(
    *,
    champion_reference_path: Optional[Path],
) -> Dict[str, Any]:
    champion_reference = resolve_champion_reference(champion_reference_path)
    if champion_reference is not None:
        return champion_reference

    raise FileNotFoundError(
        "Unable to resolve a production-safe model reference for inference. "
        "Configure a valid champion reference for the canonical production runtime."
    )
