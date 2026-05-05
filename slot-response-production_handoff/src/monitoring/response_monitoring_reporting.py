from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .contracts import DETAILS_DIRNAME, monitoring_artifact_contract


def ensure_mode_dir(root: Path, mode: str, run_label: str) -> Path:
    path = root / mode / run_label
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_frame(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def write_markdown(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_monitoring_bundle(
    *,
    output_root: Path,
    mode: str,
    run_label: str,
    markdown_lines: List[str],
    artifact_payloads: Dict[str, Any],
) -> Path:
    mode_dir = ensure_mode_dir(output_root, mode, run_label)
    details_dir = mode_dir / DETAILS_DIRNAME
    artifacts = monitoring_artifact_contract(mode)
    for filename in artifacts:
        payload = artifact_payloads[filename]
        target = details_dir / filename
        if filename.endswith(".csv"):
            write_frame(target, payload)
        elif filename.endswith(".json"):
            write_json(target, payload)
        elif filename.endswith(".md"):
            write_markdown(mode_dir / filename, payload)
            write_markdown(target, payload)
        else:
            raise ValueError(f"Unsupported artifact file type: {filename}")
    return mode_dir
