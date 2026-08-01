# Hopeful Ribbon

Hopeful Ribbon is an educational Flask application that demonstrates a reproducible binary-classification workflow on the Wisconsin Diagnostic Breast Cancer dataset. It compares four scikit-learn model families, deploys the pipeline selected by training-only stratified cross-validation, validates inference inputs, and can suggest nearby medical facilities through an optional location integration.

> **Medical disclaimer:** This repository is a machine-learning education project, not a clinical tool. Its output is not a diagnosis, treatment recommendation, or estimate of an individual's real-world cancer risk. Dataset performance does not establish medical validity. A qualified medical professional must interpret any actual biopsy or health concern.

## Architecture

```text
Explicit training command                         Flask runtime
CSV -> schema validation -> stratified split      startup -> validate metadata
    -> training-only GridSearchCV                  -> load fitted pipeline once
    -> recall-first model selection                -> validate ordered inputs
    -> one untouched test evaluation               -> class + probability
    -> model, metadata, metrics, plots              -> optional facility API
```

Training and inference are deliberately separate. `scripts/train_model.py` is the only training entry point. The Flask application factory loads `artifacts/model.joblib` and its metadata once at startup and fails clearly when the files are missing or incompatible; no request can trigger retraining.

## Dataset and target

The checked-in `App/data/breast-cancer.csv` is derived from the [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic) (DOI `10.24432/C5DW2B`). Features describe properties computed from digitized images of fine-needle aspirate cell nuclei. The target mapping is:

- `M` (malignant) -> `1`, the positive class
- `B` (benign) -> `0`

The repository copy contains 567 rows: 211 malignant and 356 benign. The canonical UCI dataset lists 569 instances, so results in this repository apply specifically to the checked-in copy. The loader removes `id` and unnamed columns, checks the target and required predictors, rejects unknown labels, and rejects missing, malformed, NaN, or infinite features.

## Features used

The deployed interface uses these exact ordered features:

1. `radius_mean`
2. `texture_mean`
3. `perimeter_mean`

They were retained to preserve the original three-field experience and keep the demonstration understandable. They were **not** selected statistically and are not claimed to be the three most important or clinically critical variables. This is interface-driven feature restriction, not dimensionality reduction.

To quantify the trade-off without touching the test set, the training pipeline compares Logistic Regression on the same training folds:

| Feature set | Features | CV malignant recall | CV F1 | CV ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Interface features | 3 | 0.8759 +/- 0.0678 | 0.8652 +/- 0.0526 | 0.9638 +/- 0.0181 |
| All numeric predictors | 30 | 0.9586 +/- 0.0398 | 0.9640 +/- 0.0187 | 0.9955 +/- 0.0045 |

The full-feature benchmark is materially stronger in cross-validation. It is retained as analysis only because the deployed UI cannot collect all 30 measurements. No full-feature result was consulted during final test evaluation.

## Training and validation methodology

- Deterministic random seed: `42`
- One stratified 80/20 train/test split
- Untouched 114-row test partition held back until final selection
- Shuffled 5-fold `StratifiedKFold` cross-validation on the 453 training rows
- `GridSearchCV` for reasonable family-specific hyperparameters
- `StandardScaler` inside the Logistic Regression and KNN pipelines
- Baseline Dummy Classifier plus Logistic Regression, KNN, and Random Forest
- Selection by mean malignant-class recall, with F1, ROC-AUC, and accuracy as tie-breakers
- Exactly one evaluation of the selected model on the test partition

This design prevents preprocessing leakage because each scaler is fitted only within its pipeline's current training fold. Candidate models never see the held-out test partition during tuning or selection.

## Model comparison

These are training-only cross-validation results for each family's recall-optimized hyperparameters:

| Model | Accuracy | Precision (M) | Recall (M) | F1 (M) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy Classifier | 0.5340 +/- 0.0367 | 0.3600 +/- 0.0573 | 0.3191 +/- 0.0483 | 0.3383 +/- 0.0525 | 0.4905 +/- 0.0396 |
| **Logistic Regression** | **0.8985 +/- 0.0389** | **0.8576 +/- 0.0580** | **0.8759 +/- 0.0678** | **0.8652 +/- 0.0526** | **0.9638 +/- 0.0181** |
| K-Nearest Neighbors | 0.9073 +/- 0.0323 | 0.8942 +/- 0.0366 | 0.8521 +/- 0.0671 | 0.8716 +/- 0.0479 | 0.9506 +/- 0.0201 |
| Random Forest | 0.8963 +/- 0.0329 | 0.8784 +/- 0.0523 | 0.8401 +/- 0.0609 | 0.8575 +/- 0.0471 | 0.9546 +/- 0.0235 |

The selected Logistic Regression uses `C=10.0` and `class_weight="balanced"`. Accuracy alone did not determine selection.

## Final untouched test results

Latest generated artifacts (`2026-08-01`, scikit-learn `1.7.2`):

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9035 |
| Malignant precision | 0.8444 |
| Malignant recall | 0.9048 |
| Malignant F1 | 0.8736 |
| ROC-AUC | 0.9732 |
| False negatives | 4 of 42 malignant test rows |
| False-negative rate | 0.0952 |

Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[65, 7], [4, 38]]`.

The four false negatives matter more than the headline accuracy: they are malignant-labelled dataset rows that the model classified as benign. In a real clinical workflow such errors could delay follow-up, which is one reason this model must not be used for patient care. Metrics are also available in `artifacts/metrics.json`; the plots are in `artifacts/confusion_matrix.png` and `artifacts/roc_curve.png`.

## Installation

Python 3.11 or newer is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Generate a secret with `python -c "import secrets; print(secrets.token_hex(32))"` and place it in `.env`.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | Yes | Flask-WTF session and CSRF protection |
| `GOOGLE_MAPS_API_KEY` | No | Google Geocoding and Places requests for nearby hospitals |
| `FACILITY_API_TIMEOUT` | No | External request timeout in seconds; default `5` |
| `MAX_CONTENT_LENGTH` | No | Maximum request body bytes; default `16384` |
| `SESSION_COOKIE_SECURE` | Production HTTPS | Sends the session cookie only over HTTPS |
| `MODEL_PATH` | No | Override the default model artifact path |
| `MODEL_METADATA_PATH` | No | Override the default metadata path |

Restrict any Maps key to only the needed APIs and deployment origins. Do not commit `.env`.

## Commands

Train and regenerate all artifacts:

```bash
python scripts/train_model.py
```

Run the tests and coverage report:

```bash
pytest
pytest --cov=App --cov-report=term-missing
```

Start the development server:

```bash
python App/app.py
```

Open `http://127.0.0.1:5001`. A production WSGI process can use `gunicorn 'App.app:app'` with a production `SECRET_KEY` and HTTPS-aware cookie configuration.

## Repository structure

```text
.
├── App/
│   ├── __init__.py              # Application factory and error handlers
│   ├── app.py                   # Local/WSGI entry point
│   ├── config.py                # Environment configuration
│   ├── routes.py                # Thin Flask routes
│   ├── forms.py                 # WTForms schema
│   ├── data/breast-cancer.csv
│   ├── ml/
│   │   ├── data.py              # Dataset validation
│   │   └── training.py          # Search, selection, evaluation, artifacts
│   ├── services/
│   │   ├── prediction.py        # Model loading and safe inference
│   │   └── facilities.py        # Timeout-bounded location integration
│   ├── templates/
│   └── static/
├── artifacts/                   # Model, metadata, metrics, and plots
├── scripts/train_model.py
├── tests/
├── .env.example
├── MODEL_CARD.md
├── SECURITY.md
├── pyproject.toml
└── requirements.txt
```

`setup_models.py` remains only as a backward-compatible wrapper; new work should call `scripts/train_model.py`.

## Limitations

- The checked-in dataset has 567 rows rather than the canonical 569 and may not match every published benchmark.
- The deployed three-feature interface gives up substantial cross-validation performance versus the 30-feature benchmark.
- This is a small, historical, curated dataset; it does not establish generalization across institutions, equipment, populations, or current clinical practice.
- No external clinical validation, probability calibration study, subgroup fairness analysis, or prospective evaluation has been performed.
- Input ranges are bounded by this dataset, not by clinical reference standards.
- Nearby facilities depend on Google provider coverage and configuration and are not endorsements.

## Future improvements

- Design a usable 30-feature import flow rather than manually collecting three values.
- Add probability calibration and decision-threshold analysis using training-only nested validation.
- Evaluate subgroup performance when an appropriately governed dataset supports it.
- Add CI for tests, formatting, dependency checks, and artifact reproducibility.
- Replace the legacy nearby-search endpoint if the configured provider retires it.

Additional intended-use and ethics details are in [MODEL_CARD.md](MODEL_CARD.md).
