from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)


# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "bipolar_artifacts"

TEST_FILE = BASE_DIR / "bipolar_test.csv"
MODEL_PATH = ARTIFACT_DIR / "bipolar_snn_model.keras"
SCALER_PATH = ARTIFACT_DIR / "bipolar_scaler.pkl"
GENES_PATH = ARTIFACT_DIR / "bipolar_selected_genes.json"
METADATA_PATH = ARTIFACT_DIR / "bipolar_metadata.json"
OUTPUT_FILE = BASE_DIR / "bipolar_test_predictions.csv"

# Contract supported by this script:
# selected genes -> optional training imputer -> training scaler -> model
# Set this ONLY if the same imputer was used before this scaler
# during SNN training, on exactly the selected genes in this order.
TRAINING_IMPUTER_PATH = None

# Use the exact mapping recorded during training, not a guess.
# Example only: {"Bipolar": 0, "Control": 1}
LABEL_MAPPING_OVERRIDE = {
    "bipolar": 0,
    "control": 1,
}


# True ONLY if numeric test labels already use the training encoding.
TEST_LABELS_ALREADY_ENCODED = False


TARGET_OVERRIDE = None
TARGET_CANDIDATES = (
    "label", "target", "class", "diagnosis", "group"
)


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def check_transformer(transformer, genes, name):
    n_features = getattr(transformer, "n_features_in_", None)
    if n_features is None:
        raise ValueError(f"{name}: missing n_features_in_; cannot verify input")
    if int(n_features) != len(genes):
        raise ValueError(
            f"{name} expects {n_features} features, but model gene list "
            f"contains {len(genes)}. Check the training pipeline."
        )

    names = getattr(transformer, "feature_names_in_", None)
    if names is not None and list(map(str, names)) != genes:
        raise ValueError(
            f"{name}: feature names/order do not match the saved model genes"
        )


def apply_transform(transformer, values, genes):
    if hasattr(transformer, "feature_names_in_"):
        values = pd.DataFrame(values, columns=genes)
    result = np.asarray(transformer.transform(values), dtype=float)
    if result.shape != (len(values), len(genes)):
        raise ValueError("Preprocessing changed the number of rows/features")
    return result


def resolve_mapping(metadata):
    mapping = LABEL_MAPPING_OVERRIDE
    if mapping is None:
        mapping = metadata.get("label_mapping")

    # Supported classes format: LabelEncoder.classes_.tolist()
    if mapping is None and metadata.get("classes") is not None:
        classes = metadata["classes"]
        if not isinstance(classes, list) or len(classes) != 2:
            raise ValueError("metadata.classes must be [class_0, class_1]")
        mapping = {str(label): i for i, label in enumerate(classes)}

    if mapping is None:
        return None
    if not isinstance(mapping, dict) or len(mapping) != 2:
        raise ValueError("label_mapping must map two original labels to 0 and 1")
    if set(mapping.values()) != {0, 1}:
        raise ValueError("label_mapping values must be exactly 0 and 1")

    normalized = {
        str(label).strip().casefold(): int(code)
        for label, code in mapping.items()
    }
    if len(normalized) != 2:
        raise ValueError("Class names become identical after normalization")
    return {str(label): int(code) for label, code in mapping.items()}


def main():
    for path in (TEST_FILE, MODEL_PATH, SCALER_PATH, GENES_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    metadata = read_json(METADATA_PATH) if METADATA_PATH.is_file() else {}
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")

    threshold = float(metadata.get("threshold", 0.5))
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold must be finite and between 0 and 1")

    genes_data = read_json(GENES_PATH)
    genes = (
        genes_data.get("selected_genes")
        if isinstance(genes_data, dict) else genes_data
    )
    if not isinstance(genes, list) or not genes:
        raise ValueError("Saved genes must be a nonempty JSON list")
    if any(not isinstance(g, str) or not g.strip() for g in genes):
        raise ValueError("Saved gene IDs must be nonempty strings")
    # Do not silently repair identifiers saved with the trained model.
    if any(g != g.strip() for g in genes):
        raise ValueError("Saved gene IDs contain whitespace; check training artifacts")
    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate genes in training artifacts")

    print(f"Reading test CSV: {TEST_FILE.resolve()}", flush=True)
    try:
        df = pd.read_csv(TEST_FILE, encoding="utf-8-sig")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Invalid CSV structure in: {TEST_FILE}") from exc
    if df.empty:
        raise ValueError("Test dataset is empty")

    missing = [g for g in genes if g not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} model features missing from test: {missing[:20]}. "
            "Do not remove rank prefixes or substitute other genes automatically."
        )

    explicit_target = TARGET_OVERRIDE or metadata.get("target_column")
    if explicit_target is not None:
        if explicit_target not in df.columns:
            raise ValueError(f"Configured target not found: {explicit_target}")
        target = explicit_target
    else:
        matches = [
            c for c in df.columns
            if str(c).strip().casefold() in TARGET_CANDIDATES
        ]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous target columns: {matches}")
        target = matches[0] if matches else None
    if target in genes:
        raise ValueError("Target column is included in model features")

    # Invalid text raises an error instead of silently becoming missing data.
    X_df = df.loc[:, genes].apply(pd.to_numeric, errors="raise")
    X = X_df.to_numpy(dtype=float)
    if np.isinf(X).any():
        raise ValueError("Test features contain infinity")

    scaler = joblib.load(SCALER_PATH)
    check_transformer(scaler, genes, "Scaler")

    if TRAINING_IMPUTER_PATH is not None:
        imputer_path = Path(TRAINING_IMPUTER_PATH)
        if not imputer_path.is_absolute():
            imputer_path = BASE_DIR / imputer_path
        if not imputer_path.is_file():
            raise FileNotFoundError(imputer_path)
        imputer = joblib.load(imputer_path)
        check_transformer(imputer, genes, "Imputer")
        X = apply_transform(imputer, X, genes)
    elif np.isnan(X).any():
        raise ValueError(
            "Missing test values found, but no verified SNN training imputer "
            "was configured. Do not fit an imputer on the test set."
        )

    if not np.isfinite(X).all():
        raise ValueError("Non-finite features remain before scaling")
    X_scaled = apply_transform(scaler, X, genes)
    if not np.isfinite(X_scaled).all():
        raise ValueError("Non-finite features after scaling")

    model = load_model(MODEL_PATH, compile=False)
    shape = model.input_shape
    if isinstance(shape, list) or len(shape) != 2:
        raise ValueError("Expected a single-input, tabular SNN model")
    if shape[-1] != len(genes):
        raise ValueError(
            f"Model expects {shape[-1]} features, received {len(genes)}"
        )

    probabilities = np.asarray(model.predict(X_scaled, verbose=0))
    if probabilities.shape != (len(df), 1):
        raise ValueError("Expected one sigmoid probability per test sample")
    y_prob = probabilities[:, 0]
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Model outputs are not valid probabilities")
    y_pred = (y_prob >= threshold).astype(int)

    mapping = resolve_mapping(metadata)
    inverse = {code: label for label, code in mapping.items()} if mapping else None

    results = df.copy()
    results["probability_encoded_class_1"] = y_prob
    results["predicted_label"] = y_pred
    if inverse is not None:
        results["predicted_class"] = [inverse[int(v)] for v in y_pred]

    results.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Samples: {len(df)} | Features: {len(genes)}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Predictions saved: {OUTPUT_FILE}")

    if target is None:
        print("No target column: predictions only; no evaluation performed.")
        return

    raw = df[target]
    if raw.isna().any():
        raise ValueError("Missing true labels; evaluation stopped")

    if TEST_LABELS_ALREADY_ENCODED:
        numeric = pd.to_numeric(raw, errors="raise")
        if not numeric.isin([0, 1]).all():
            raise ValueError("Encoded labels must contain only 0 and 1")
        y_true = numeric.to_numpy(dtype=int)
    else:
        if mapping is None:
            raise ValueError(
                "Numeric predictions were saved, but evaluation requires the "
                "original training label_mapping or LabelEncoder classes. "
                "Set LABEL_MAPPING_OVERRIDE only after verifying that mapping."
            )
        normalized_mapping = {
            k.strip().casefold(): v for k, v in mapping.items()
        }
        normalized = raw.astype(str).str.strip().str.casefold()
        unknown = sorted(set(normalized) - set(normalized_mapping))
        if unknown:
            raise ValueError(f"Labels absent from training mapping: {unknown}")
        y_true = normalized.map(normalized_mapping).to_numpy(dtype=int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    auc = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) == 2 else np.nan
    )

    names = [inverse[0], inverse[1]] if inverse else ["Class 0", "Class 1"]
    print("\nFINAL TEST RESULTS")
    print(f"Positive class: {names[1]} (encoded 1)")
    print(f"Accuracy    : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision   : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall      : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1          : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC     : {auc:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print("Confusion matrix: rows=true, columns=predicted; order=[0, 1]")
    print(matrix)
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=names, zero_division=0
    ))

    results["actual_label"] = y_true
    if inverse is not None:
        results["actual_class"] = [inverse[int(v)] for v in y_true]
    results["correct_prediction"] = y_true == y_pred
    results.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Final results saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
