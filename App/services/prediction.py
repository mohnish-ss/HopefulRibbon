"""Validated, artifact-backed model inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

INPUT_TO_FEATURE = {
    "radius": "radius_mean",
    "texture": "texture_mean",
    "perimeter": "perimeter_mean",
}
EXPECTED_METADATA_VERSION = 1


class ArtifactError(RuntimeError):
    """Raised when model artifacts are absent or incompatible."""


class InputValidationError(ValueError):
    """Raised for unsafe or incomplete prediction inputs."""


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: str
    display_label: str
    malignant_probability: float
    confidence: float

    @property
    def cautious_message(self) -> str:
        comparison = "malignant" if self.predicted_label == "M" else "benign"
        return (
            "The model classified the submitted measurements as more similar to "
            f"the {comparison} class in the training dataset."
        )


class PredictionService:
    """Loads a fitted pipeline and enforces its metadata contract."""

    def __init__(self, model: Pipeline, metadata: dict[str, object]) -> None:
        self.model = model
        self.metadata = metadata
        self.feature_names = list(metadata["feature_names"])
        self.feature_ranges = dict(metadata["feature_ranges"])

    @classmethod
    def load(cls, model_path: Path | str, metadata_path: Path | str):
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        missing = [
            str(path) for path in (model_path, metadata_path) if not path.is_file()
        ]
        if missing:
            raise ArtifactError(
                "Model artifacts are missing: "
                + ", ".join(missing)
                + ". Run `python scripts/train_model.py`."
            )

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(
                f"Model metadata is unreadable: {metadata_path}"
            ) from error

        cls._validate_metadata(metadata)
        try:
            model = joblib.load(model_path)
        except Exception as error:
            raise ArtifactError(
                f"Model artifact is unreadable: {model_path}"
            ) from error
        if not isinstance(model, Pipeline):
            raise ArtifactError(
                "Model artifact must contain a fitted sklearn Pipeline."
            )
        return cls(model, metadata)

    @staticmethod
    def _validate_metadata(metadata: dict[str, object]) -> None:
        required = {
            "artifact_format_version",
            "feature_names",
            "feature_ranges",
            "target_mapping",
            "scikit_learn_version",
            "final_test_metrics",
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise ArtifactError(f"Model metadata is missing: {', '.join(missing)}")
        if metadata["artifact_format_version"] != EXPECTED_METADATA_VERSION:
            raise ArtifactError("Unsupported model artifact format version.")
        features = metadata["feature_names"]
        if not isinstance(features, list) or features != list(
            INPUT_TO_FEATURE.values()
        ):
            raise ArtifactError(
                "Model metadata feature order is incompatible with this app."
            )
        trained_version = str(metadata["scikit_learn_version"])
        if trained_version.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            raise ArtifactError(
                "Model was trained with scikit-learn "
                f"{trained_version}, but runtime version is {sklearn.__version__}. "
                "Retrain the artifacts with the installed dependencies."
            )

    @property
    def public_summary(self) -> dict[str, object]:
        return {
            "model_type": self.metadata["model_type"],
            "test_metrics": self.metadata["final_test_metrics"],
            "feature_names": self.feature_names,
            "training_date": self.metadata["training_date"],
            "dataset_rows": self.metadata["dataset_rows"],
        }

    def parse_inputs(self, form_data: Mapping[str, object]) -> pd.DataFrame:
        values: dict[str, float] = {}
        for input_name, feature_name in INPUT_TO_FEATURE.items():
            raw_value = form_data.get(input_name)
            if raw_value is None or str(raw_value).strip() == "":
                raise InputValidationError(f"{input_name.title()} is required.")
            try:
                value = float(str(raw_value).strip())
            except (TypeError, ValueError) as error:
                raise InputValidationError(
                    f"{input_name.title()} must be a number."
                ) from error
            if not math.isfinite(value):
                raise InputValidationError(
                    f"{input_name.title()} must be a finite number."
                )
            bounds = self.feature_ranges.get(feature_name)
            if not isinstance(bounds, dict):
                raise ArtifactError(f"Missing validation range for {feature_name}.")
            minimum = float(bounds["min"])
            maximum = float(bounds["max"])
            if not minimum <= value <= maximum:
                raise InputValidationError(
                    f"{input_name.title()} must be between {minimum:g} and {maximum:g}."
                )
            values[feature_name] = value
        return pd.DataFrame(
            [[values[name] for name in self.feature_names]], columns=self.feature_names
        )

    def predict(self, form_data: Mapping[str, object]) -> PredictionResult:
        frame = self.parse_inputs(form_data)
        prediction = int(self.model.predict(frame)[0])
        if not hasattr(self.model, "predict_proba"):
            raise ArtifactError("The deployed pipeline does not support probabilities.")
        probabilities = self.model.predict_proba(frame)[0]
        classes = list(self.model.classes_)
        malignant_probability = float(probabilities[classes.index(1)])
        predicted_probability = float(probabilities[classes.index(prediction)])
        label = "M" if prediction == 1 else "B"
        return PredictionResult(
            predicted_label=label,
            display_label="Malignant" if label == "M" else "Benign",
            malignant_probability=round(malignant_probability, 4),
            confidence=round(predicted_probability, 4),
        )
