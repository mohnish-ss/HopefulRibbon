from __future__ import annotations

from pathlib import Path

import pytest

from App.services.prediction import (
    ArtifactError,
    InputValidationError,
    PredictionService,
)
from tests.conftest import ARTIFACTS_PATH


@pytest.fixture()
def service() -> PredictionService:
    return PredictionService.load(
        ARTIFACTS_PATH / "model.joblib", ARTIFACTS_PATH / "model_metadata.json"
    )


def test_prediction_input_order_matches_artifact(service):
    frame = service.parse_inputs(
        {"perimeter": "122.8", "radius": "17.99", "texture": "10.38"}
    )
    assert frame.columns.tolist() == ["radius_mean", "texture_mean", "perimeter_mean"]
    assert frame.iloc[0].tolist() == [17.99, 10.38, 122.8]


@pytest.mark.parametrize("missing", ["radius", "texture", "perimeter"])
def test_missing_prediction_input_is_rejected(service, missing):
    values = {"radius": "17.99", "texture": "10.38", "perimeter": "122.8"}
    values.pop(missing)
    with pytest.raises(InputValidationError, match="required"):
        service.predict(values)


def test_nonnumeric_prediction_input_is_rejected(service):
    with pytest.raises(InputValidationError, match="must be a number"):
        service.predict({"radius": "abc", "texture": "10.38", "perimeter": "122.8"})


@pytest.mark.parametrize("value", ["-1", "1000", "nan", "inf"])
def test_out_of_range_or_nonfinite_input_is_rejected(service, value):
    with pytest.raises(InputValidationError):
        service.predict({"radius": value, "texture": "10.38", "perimeter": "122.8"})


def test_model_loading_fails_clearly_when_artifacts_are_missing(tmp_path):
    with pytest.raises(ArtifactError, match="Run `python scripts/train_model.py`"):
        PredictionService.load(tmp_path / "model.joblib", tmp_path / "metadata.json")


def test_model_loading_succeeds(service):
    assert service.feature_names == ["radius_mean", "texture_mean", "perimeter_mean"]
