"""Reproducible model comparison, selection, and final evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

_cache_root = Path(tempfile.gettempdir()) / "hopeful-ribbon-plot-cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import DEFAULT_FEATURES, TARGET_MAPPING, load_dataset

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    pipeline: Pipeline
    parameters: dict[str, list[Any]]


def candidate_models(random_seed: int = RANDOM_SEED) -> list[Candidate]:
    """Return bounded, interview-readable searches for four model families."""
    return [
        Candidate(
            "Dummy Classifier",
            Pipeline([("classifier", DummyClassifier(random_state=random_seed))]),
            {"classifier__strategy": ["most_frequent", "stratified"]},
        ),
        Candidate(
            "Logistic Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(max_iter=5000, random_state=random_seed),
                    ),
                ]
            ),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        Candidate(
            "K-Nearest Neighbors",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier()),
                ]
            ),
            {
                "classifier__n_neighbors": [3, 5, 7, 9, 13, 17],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        ),
        Candidate(
            "Random Forest",
            Pipeline(
                [
                    (
                        "classifier",
                        RandomForestClassifier(
                            random_state=random_seed,
                            n_jobs=1,
                        ),
                    )
                ]
            ),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 6, 12],
                "classifier__min_samples_leaf": [1, 3],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    ]


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the only train/test split used by the training run."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=random_seed,
        stratify=target,
    )


def compare_candidates(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int = RANDOM_SEED,
    candidates: list[Candidate] | None = None,
) -> tuple[Pipeline, str, list[dict[str, Any]]]:
    """Tune only on training folds and select primarily by malignant recall."""
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=random_seed
    )
    comparisons: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for candidate in candidates or candidate_models(random_seed):
        search = GridSearchCV(
            estimator=candidate.pipeline,
            param_grid=candidate.parameters,
            scoring=SCORING,
            refit="recall",
            cv=cross_validation,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        best_index = search.best_index_
        metrics = {
            metric: {
                "mean": round(
                    float(search.cv_results_[f"mean_test_{metric}"][best_index]), 6
                ),
                "std": round(
                    float(search.cv_results_[f"std_test_{metric}"][best_index]), 6
                ),
            }
            for metric in SCORING
        }
        comparisons.append(
            {
                "model": candidate.name,
                "cross_validation": metrics,
                "selected_hyperparameters": _json_safe_parameters(search.best_params_),
            }
        )
        fitted[candidate.name] = search.best_estimator_

    winner = max(
        comparisons,
        key=lambda row: (
            row["cross_validation"]["recall"]["mean"],
            row["cross_validation"]["f1"]["mean"],
            row["cross_validation"]["roc_auc"]["mean"],
            row["cross_validation"]["accuracy"]["mean"],
        ),
    )
    return fitted[winner["model"]], winner["model"], comparisons


def train_and_evaluate(
    dataset_path: Path | str,
    artifacts_dir: Path | str,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Train, select with CV, evaluate once, and persist all artifacts."""
    dataset_path = Path(dataset_path)
    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)

    features, target = load_dataset(dataset_path, DEFAULT_FEATURES)
    all_features, all_target = load_dataset(dataset_path, None)
    x_train, x_test, y_train, y_test = split_dataset(features, target, random_seed)

    selected_model, selected_name, comparisons = compare_candidates(
        x_train, y_train, random_seed
    )
    winner_comparison = next(
        row for row in comparisons if row["model"] == selected_name
    )

    full_feature_comparison = _compare_interface_feature_tradeoff(
        x_train=x_train,
        y_train=y_train,
        all_features=all_features,
        all_target=all_target,
        random_seed=random_seed,
        deployed_comparisons=comparisons,
    )

    # The untouched test partition is consulted only here, after selection is final.
    predictions = selected_model.predict(x_test)
    probabilities = selected_model.predict_proba(x_test)[
        :, list(selected_model.classes_).index(1)
    ]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    positive_count = false_negative + true_positive
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "precision_malignant": round(float(precision_score(y_test, predictions)), 6),
        "recall_malignant": round(float(recall_score(y_test, predictions)), 6),
        "f1_malignant": round(float(f1_score(y_test, predictions)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "false_negative_count": int(false_negative),
        "false_negative_rate": round(float(false_negative / positive_count), 6),
        "confusion_matrix": {
            "labels": ["Benign", "Malignant"],
            "values": matrix.tolist(),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }
    distribution = {
        "all": _class_distribution(target),
        "train": _class_distribution(y_train),
        "test": _class_distribution(y_test),
    }
    feature_ranges = {
        name: {
            "min": float(features[name].min()),
            "max": float(features[name].max()),
        }
        for name in DEFAULT_FEATURES
    }
    training_date = datetime.now(timezone.utc).isoformat()
    metadata = {
        "artifact_format_version": 1,
        "model_type": selected_name,
        "feature_names": DEFAULT_FEATURES,
        "feature_ranges": feature_ranges,
        "feature_rationale": (
            "The three mean measurements are retained for compatibility with the "
            "existing educational UI, not because they were proven to be the most important."
        ),
        "target_mapping": TARGET_MAPPING,
        "positive_class": "M",
        "training_date": training_date,
        "scikit_learn_version": sklearn.__version__,
        "dataset_filename": dataset_path.name,
        "dataset_rows": int(len(features)),
        "random_seed": random_seed,
        "validation_strategy": (
            f"Stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} train/test split; "
            f"{CV_FOLDS}-fold shuffled StratifiedKFold GridSearchCV on training data only; "
            "model family selected primarily by malignant recall; one final test evaluation."
        ),
        "selected_hyperparameters": winner_comparison["selected_hyperparameters"],
        "selected_cross_validation": winner_comparison["cross_validation"],
        "class_distribution": distribution,
        "final_test_metrics": metrics,
    }
    report = {
        "generated_at": training_date,
        "selection_rule": (
            "Highest mean cross-validation recall for malignant class; ties broken by "
            "F1, ROC-AUC, then accuracy."
        ),
        "selected_model": selected_name,
        "model_comparison": comparisons,
        "feature_set_comparison_training_cv_only": full_feature_comparison,
        "test_metrics": metrics,
        "class_distribution": distribution,
        "features_used": DEFAULT_FEATURES,
        "selected_hyperparameters": winner_comparison["selected_hyperparameters"],
    }

    joblib.dump(selected_model, output / "model.joblib")
    _write_json(output / "model_metadata.json", metadata)
    _write_json(output / "metrics.json", report)
    _save_confusion_matrix(matrix, output / "confusion_matrix.png")
    _save_roc_curve(y_test, probabilities, metrics["roc_auc"], output / "roc_curve.png")
    return report


def _compare_interface_feature_tradeoff(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    all_features: pd.DataFrame,
    all_target: pd.Series,
    random_seed: int,
    deployed_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    logistic = next(
        c for c in candidate_models(random_seed) if c.name == "Logistic Regression"
    )
    full_train = all_features.loc[x_train.index]
    if not all_target.loc[x_train.index].equals(y_train):
        raise RuntimeError("Full and interface feature targets are misaligned.")
    _, _, full_results = compare_candidates(
        full_train, y_train, random_seed, candidates=[logistic]
    )
    three_feature = next(
        row for row in deployed_comparisons if row["model"] == "Logistic Regression"
    )
    return {
        "method": "Logistic Regression GridSearchCV on identical training folds",
        "three_interface_features": three_feature,
        "all_numeric_predictor_features": {
            "feature_count": int(all_features.shape[1]),
            **full_results[0],
        },
        "note": "No final test data was used for this feature-set comparison.",
    }


def _class_distribution(target: pd.Series) -> dict[str, dict[str, float | int]]:
    total = len(target)
    counts = target.value_counts().to_dict()
    return {
        "benign": {
            "count": int(counts.get(0, 0)),
            "proportion": round(counts.get(0, 0) / total, 6),
        },
        "malignant": {
            "count": int(counts.get(1, 0)),
            "proportion": round(counts.get(1, 0) / total, 6),
        },
    }


def _json_safe_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key.removeprefix("classifier__"): value for key, value in parameters.items()
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Malignant"]).plot(
        ax=axis, cmap="Blues", colorbar=False
    )
    axis.set_title("Untouched test-set confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_roc_curve(
    target: pd.Series, probabilities: np.ndarray, roc_auc: float, path: Path
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(target, probabilities)
    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    axis.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {roc_auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#667085", label="Chance")
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="Untouched test-set ROC curve",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
