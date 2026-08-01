"""Strict loading and validation for the Wisconsin dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

TARGET_COLUMN = "diagnosis"
TARGET_MAPPING = {"B": 0, "M": 1}
DEFAULT_FEATURES = ["radius_mean", "texture_mean", "perimeter_mean"]


class DatasetValidationError(ValueError):
    """Raised when training data violates the expected schema."""


def load_dataset(
    dataset_path: Path | str,
    feature_names: Sequence[str] | None = DEFAULT_FEATURES,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load numeric predictors and map malignant to positive class 1."""
    path = Path(dataset_path)
    if not path.is_file():
        raise DatasetValidationError(f"Dataset not found: {path}")
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise DatasetValidationError(f"Dataset could not be read: {path}") from error
    if data.empty:
        raise DatasetValidationError("Dataset is empty.")

    data.columns = [str(column).strip() for column in data.columns]
    data = data.drop(columns=[c for c in data if c.lower().startswith("unnamed")])
    if TARGET_COLUMN not in data.columns:
        raise DatasetValidationError(
            f"Required target column is missing: {TARGET_COLUMN}"
        )

    labels = data[TARGET_COLUMN].astype("string").str.strip().str.upper()
    unknown_labels = sorted(set(labels.dropna()) - TARGET_MAPPING.keys())
    if unknown_labels:
        raise DatasetValidationError(
            "Target contains unsupported labels: " + ", ".join(unknown_labels)
        )
    target = labels.map(TARGET_MAPPING)
    if target.isna().any():
        raise DatasetValidationError("Target contains missing or malformed values.")
    target = target.astype("int64").rename(TARGET_COLUMN)
    if set(target.unique()) != {0, 1}:
        raise DatasetValidationError(
            "Target must contain both benign and malignant rows."
        )

    if feature_names is None:
        selected = [
            column for column in data.columns if column not in {TARGET_COLUMN, "id"}
        ]
    else:
        selected = list(feature_names)
    missing = [column for column in selected if column not in data.columns]
    if missing:
        raise DatasetValidationError(
            "Required feature columns are missing: " + ", ".join(missing)
        )
    if not selected:
        raise DatasetValidationError("No predictor columns were selected.")

    features = data.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    invalid_columns = features.columns[features.isna().any()].tolist()
    if invalid_columns:
        raise DatasetValidationError(
            "Features contain missing or malformed values: "
            + ", ".join(invalid_columns)
        )
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise DatasetValidationError("Features contain infinite values.")
    return features.astype("float64"), target
