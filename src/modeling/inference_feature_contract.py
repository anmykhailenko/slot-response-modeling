from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd


def resolve_expected_raw_feature_order(
    *,
    declared_feature_columns: Sequence[str],
    preprocessor: Any,
) -> List[str]:
    expected = [str(column) for column in declared_feature_columns]
    fitted_feature_names_in = [str(column) for column in getattr(preprocessor, "feature_names_in_", [])]
    if not expected:
        raise ValueError("Inference feature contract requires a non-empty declared feature list.")
    if not fitted_feature_names_in:
        raise ValueError("Saved preprocessor is missing `feature_names_in_`; cannot validate the raw inference feature contract.")
    if fitted_feature_names_in != expected:
        raise ValueError(
            "Saved preprocessor raw feature contract does not match the declared scoring feature order. "
            f"declared={expected}, preprocessor={fitted_feature_names_in}"
        )
    return expected


def validate_scoring_input_frame(
    scoring_input: pd.DataFrame,
    *,
    expected_raw_feature_order: Sequence[str],
) -> None:
    actual = [str(column) for column in scoring_input.columns]
    expected = [str(column) for column in expected_raw_feature_order]
    missing = [column for column in expected if column not in actual]
    unexpected = [column for column in actual if column not in expected]
    if missing or unexpected:
        raise ValueError(
            "Scoring dataset raw feature columns do not match the trained model contract. "
            f"missing={missing}, unexpected={unexpected}"
        )
    if actual != expected:
        raise ValueError(
            "Scoring dataset raw feature order does not match the trained model contract. "
            f"expected={expected}, actual={actual}"
        )


def resolve_expected_transformed_feature_names(
    *,
    declared_transformed_feature_names: Sequence[str],
    preprocessor: Any,
    estimator: Any,
) -> List[str]:
    declared = [str(name) for name in declared_transformed_feature_names]
    preprocessor_feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    stripped_preprocessor_feature_names = [str(name).split("__", 1)[-1] for name in preprocessor_feature_names]
    if not declared:
        declared = preprocessor_feature_names
    if declared != preprocessor_feature_names and declared != stripped_preprocessor_feature_names:
        raise ValueError(
            "Declared transformed feature names do not match the fitted preprocessor output. "
            f"declared={declared}, preprocessor={preprocessor_feature_names}"
        )

    estimator_feature_names_in = [str(name) for name in getattr(estimator, "feature_names_in_", [])]
    estimator_feature_names = [str(name) for name in getattr(estimator, "feature_name_", [])]
    scoring_feature_names = estimator_feature_names_in or estimator_feature_names
    if scoring_feature_names:
        expected_feature_names = preprocessor_feature_names
        if len(scoring_feature_names) != len(expected_feature_names):
            raise ValueError(
                "Estimator transformed feature width does not match the declared scoring feature contract. "
                f"estimator={len(scoring_feature_names)}, declared={len(expected_feature_names)}"
            )
        generic_names = [f"Column_{index}" for index in range(len(expected_feature_names))]
        if scoring_feature_names != generic_names and scoring_feature_names != expected_feature_names:
            raise ValueError(
                "Estimator transformed feature names do not match the expected scoring feature contract. "
                f"estimator={scoring_feature_names}, expected={expected_feature_names}"
            )
        return scoring_feature_names
    return preprocessor_feature_names


def build_named_transformed_frame(
    transformed: Any,
    *,
    transformed_feature_names: Sequence[str],
    index: pd.Index,
) -> pd.DataFrame:
    if isinstance(transformed, pd.DataFrame):
        frame = transformed.copy()
        if frame.columns.tolist() != list(transformed_feature_names):
            raise ValueError(
                "Preprocessor returned a DataFrame whose columns do not match the transformed feature contract. "
                f"expected={list(transformed_feature_names)}, actual={frame.columns.tolist()}"
            )
    elif hasattr(transformed, "toarray"):
        frame = pd.DataFrame(transformed.toarray(), columns=list(transformed_feature_names), index=index)
    else:
        frame = pd.DataFrame(transformed, columns=list(transformed_feature_names), index=index)

    if frame.columns.tolist() != list(transformed_feature_names):
        raise ValueError("Named transformed feature frame is not aligned to the expected transformed feature order.")

    bad_columns: List[str] = []
    for column in frame.columns:
        if str(frame[column].dtype) == "object":
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except Exception:
                bad_columns.append(column)
    if bad_columns:
        raise ValueError(
            "Transformed feature frame contains non-numeric columns after preprocessing: "
            + ", ".join(bad_columns)
        )
    return frame


def format_feature_preview(feature_names: Sequence[str], preview_size: int = 5) -> Dict[str, Any]:
    names = [str(name) for name in feature_names]
    return {
        "feature_count": len(names),
        "feature_preview": names[: max(0, int(preview_size))],
    }
