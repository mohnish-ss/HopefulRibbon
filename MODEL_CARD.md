# Hopeful Ribbon model card

## Model details

- **Model:** scikit-learn Logistic Regression pipeline with `StandardScaler`
- **Artifact:** `artifacts/model.joblib`
- **Version evidence:** `artifacts/model_metadata.json`
- **Positive class:** malignant (`M` -> `1`)
- **Inputs:** ordered mean radius, mean texture, and mean perimeter
- **Selection:** highest mean malignant recall in training-only stratified five-fold cross-validation, with F1, ROC-AUC, and accuracy tie-breakers

## Intended use

The model is intended to demonstrate reproducible classification, leakage-resistant preprocessing, model comparison, serialized Flask inference, input validation, and error analysis for students, recruiters, and technical interviewers.

## Out-of-scope use

Do not use this model to diagnose, screen, triage, reassure, treat, or make care decisions for any person. Do not use its probabilities as clinical risk estimates. It is not validated for deployment in a healthcare setting.

## Data

The repository includes a 567-row copy derived from the UCI Breast Cancer Wisconsin (Diagnostic) dataset. The target contains 356 benign and 211 malignant rows. Identifier and unnamed columns are excluded. The canonical source describes 569 instances and 30 numeric predictors; therefore the checked-in copy is the authoritative input for this repository's reported results.

The deployed feature subset exists for compatibility with the original interface, not because the three variables were established as most important. A training-only comparison records the performance trade-off against all 30 numeric predictors.

## Evaluation

The deterministic protocol uses a stratified 80/20 split and shuffled five-fold stratified grid search on training data only. Logistic Regression was selected with `C=10.0` and balanced class weights.

Untouched test results on 114 rows:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9035 |
| Malignant precision | 0.8444 |
| Malignant recall | 0.9048 |
| Malignant F1 | 0.8736 |
| ROC-AUC | 0.9732 |
| False negatives | 4 |
| False-negative rate | 0.0952 |

The confusion matrix is `[[65, 7], [4, 38]]`, ordered as `[[TN, FP], [FN, TP]]`. Accuracy is insufficient on its own: four malignant-labelled test rows were missed.

## Limitations and ethical considerations

- Dataset labels and measurements are not a substitute for clinical assessment.
- Performance on a small historical dataset may not transfer across demographics, laboratories, scanners, or care settings.
- The repository dataset is two rows smaller than the canonical source.
- No clinical, prospective, subgroup fairness, robustness, or probability-calibration evaluation has been performed.
- The three-feature application underperforms a training-only 30-feature Logistic Regression benchmark.
- A polished interface can create unjustified trust. The application therefore uses cautious class-similarity wording and repeats its disclaimer with every result.
- Location suggestions are informational provider search results, not medical recommendations or endorsements.

## Maintenance

Retrain with `python scripts/train_model.py` after any data, dependency, feature, or search-space change. Review `metrics.json`, both plots, metadata compatibility, tests, and user-facing claims before publishing updated artifacts.
