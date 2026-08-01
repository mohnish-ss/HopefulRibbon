from __future__ import annotations

import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from App.ml.data import load_dataset
from App.ml.training import (
    Candidate,
    compare_candidates,
    split_dataset,
    train_and_evaluate,
)
from tests.conftest import DATASET_PATH


def test_stratified_split_is_reproducible():
    features, target = load_dataset(DATASET_PATH)
    first = split_dataset(features, target, random_seed=42)
    second = split_dataset(features, target, random_seed=42)
    for first_part, second_part in zip(first, second, strict=True):
        assert first_part.equals(second_part)
    assert first[3].mean() == second[3].mean()


def test_model_search_is_reproducible_with_fixed_seed():
    features, target = load_dataset(DATASET_PATH)
    x_train, _x_test, y_train, _y_test = split_dataset(features, target)
    candidate = Candidate(
        "Logistic Regression",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        {"classifier__C": [0.1, 1.0], "classifier__class_weight": [None, "balanced"]},
    )
    first_model, first_name, first_results = compare_candidates(
        x_train, y_train, candidates=[candidate]
    )
    second_model, second_name, second_results = compare_candidates(
        x_train, y_train, candidates=[candidate]
    )
    assert first_name == second_name
    assert first_results == second_results
    assert (first_model.predict(x_train) == second_model.predict(x_train)).all()


def test_training_creates_complete_model_artifacts(tmp_path):
    report = train_and_evaluate(DATASET_PATH, tmp_path)
    expected = {
        "model.joblib",
        "model_metadata.json",
        "metrics.json",
        "confusion_matrix.png",
        "roc_curve.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert isinstance(joblib.load(tmp_path / "model.joblib"), Pipeline)
    metadata = json.loads((tmp_path / "model_metadata.json").read_text())
    assert metadata["positive_class"] == "M"
    assert metadata["feature_names"] == [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
    ]
    assert report["test_metrics"]["false_negative_count"] >= 0
