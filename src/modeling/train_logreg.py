from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


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
                        ("scaler", StandardScaler()),
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
        raise ValueError("Logistic regression preprocessing requires at least one feature column.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_estimator(model_config: Dict[str, Any], random_seed: int) -> LogisticRegression:
    return LogisticRegression(
        solver=model_config.get("solver", "liblinear"),
        max_iter=int(model_config.get("max_iter", 1000)),
        class_weight=model_config.get("class_weight", "balanced"),
        C=float(model_config.get("C", 1.0)),
        random_state=random_seed,
    )


def feature_importance_frame(estimator: LogisticRegression, fitted_feature_names: List[str]) -> pd.DataFrame:
    coefficients = np.asarray(estimator.coef_).ravel()
    if coefficients.shape[0] != len(fitted_feature_names):
        raise ValueError(
            f"LogisticRegression coefficient count does not match feature count: "
            f"{coefficients.shape[0]} coefficients vs {len(fitted_feature_names)} fitted features."
        )
    frame = pd.DataFrame(
        {
            "feature": fitted_feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "odds_ratio": np.exp(coefficients),
        }
    )
    return frame.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def feature_importance_frames(estimator: LogisticRegression, fitted_feature_names: List[str]) -> Dict[str, pd.DataFrame]:
    return {"coefficient": feature_importance_frame(estimator, fitted_feature_names)}
