from __future__ import annotations

import pandas as pd
import pytest

from App.ml.data import (
    DEFAULT_FEATURES,
    DatasetValidationError,
    load_dataset,
)
from tests.conftest import DATASET_PATH


def test_dataset_loading_removes_identifiers_and_unnamed_columns():
    features, target = load_dataset(DATASET_PATH, None)
    assert "id" not in features.columns
    assert not any(name.lower().startswith("unnamed") for name in features.columns)
    assert len(features) == len(target) == 567
    assert features.shape[1] == 30


def test_required_column_validation(tmp_path):
    data = pd.read_csv(DATASET_PATH).drop(columns=[DEFAULT_FEATURES[0]])
    path = tmp_path / "missing.csv"
    data.to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="Required feature columns"):
        load_dataset(path)


def test_target_mapping_uses_malignant_as_positive_class():
    _features, target = load_dataset(DATASET_PATH)
    assert set(target.unique()) == {0, 1}
    assert int((target == 1).sum()) == 211
    assert int((target == 0).sum()) == 356


def test_unknown_target_label_is_rejected(tmp_path):
    data = pd.read_csv(DATASET_PATH)
    data.loc[0, "diagnosis"] = "unknown"
    path = tmp_path / "bad-target.csv"
    data.to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="unsupported labels"):
        load_dataset(path)


def test_malformed_numeric_value_is_rejected(tmp_path):
    data = pd.read_csv(DATASET_PATH)
    data["radius_mean"] = data["radius_mean"].astype("object")
    data.loc[0, "radius_mean"] = "not-a-number"
    path = tmp_path / "bad-number.csv"
    data.to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="missing or malformed"):
        load_dataset(path)
