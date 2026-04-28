from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    LGBMClassifier = Any  # type: ignore
    _LIGHTGBM_IMPORT_ERROR = exc


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_feature_columns: List[str], categorical_feature_columns: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_feature_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=0.0,
                                keep_empty_features=True,
                            ),
                        ),
                    ]
                ),
                list(numeric_feature_columns),
            )
        )
    if categorical_feature_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__missing__",
                                keep_empty_features=True,
                            ),
                        ),
                        ("encoder", _build_one_hot_encoder()),
                    ]
                ),
                list(categorical_feature_columns),
            )
        )
    if not transformers:
        raise ValueError("LightGBM preprocessing requires at least one feature column.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_estimator(model_config: Dict[str, Any], random_seed: int) -> LGBMClassifier:
    if _LIGHTGBM_IMPORT_ERROR is not None:
        raise ImportError(
            "LightGBM is enabled but could not be imported. Check the local LightGBM/runtime installation."
        ) from _LIGHTGBM_IMPORT_ERROR
    return LGBMClassifier(
        objective="binary",
        random_state=random_seed,
        n_estimators=int(model_config.get("n_estimators", 200)),
        learning_rate=float(model_config.get("learning_rate", 0.05)),
        num_leaves=int(model_config.get("num_leaves", 31)),
        min_child_samples=int(model_config.get("min_child_samples", model_config.get("min_data_in_leaf", 20))),
        subsample=float(model_config.get("subsample", 1.0)),
        colsample_bytree=float(model_config.get("colsample_bytree", 1.0)),
        reg_alpha=float(model_config.get("reg_alpha", 0.0)),
        reg_lambda=float(model_config.get("reg_lambda", 0.0)),
        class_weight=model_config.get("class_weight"),
        scale_pos_weight=model_config.get("scale_pos_weight"),
        n_jobs=-1,
        verbosity=-1,
    )


def feature_importance_frames(estimator: Any, feature_columns: List[str]) -> Dict[str, pd.DataFrame]:
    gain_values = estimator.booster_.feature_importance(importance_type="gain")
    split_values = estimator.booster_.feature_importance(importance_type="split")
    if len(gain_values) != len(feature_columns) or len(split_values) != len(feature_columns):
        raise ValueError(
            "LightGBM feature importance width does not match fitted feature names: "
            f"gain={len(gain_values)} split={len(split_values)} features={len(feature_columns)}"
        )
    gain_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": gain_values,
        }
    ).sort_values(["importance_gain", "feature"], ascending=[False, True]).reset_index(drop=True)
    split_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_split": split_values,
        }
    ).sort_values(["importance_split", "feature"], ascending=[False, True]).reset_index(drop=True)
    return {"gain": gain_frame, "split": split_frame}


def feature_importance_frame(estimator: Any, feature_columns: List[str]) -> pd.DataFrame:
    frames = feature_importance_frames(estimator, feature_columns)
    return frames["gain"].merge(frames["split"], on="feature", how="inner")
