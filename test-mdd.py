#!/usr/bin/env python3
"""Evaluate a saved MDD SNN on an independent, labeled test CSV.

Usage:
    python test-mdd.py [--test mdd_test.csv] [--artifacts-dir mdd_model_artifacts]
                       [--output-dir mdd_test_results] [--target-column COLUMN]
                       [--batch-size 32]

Dependencies: numpy, pandas, scikit-learn, joblib, and TensorFlow (for inference).
This script only loads saved preprocessing/model artifacts; it never fits or trains.
The classification threshold is fixed at probability >= 0.5. Pickle artifacts can
execute code when loaded: only use artifacts from a trusted source.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

from sklearn.metrics import matthews_corrcoef


import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
)


ROOT = Path(__file__).resolve().parent
THRESHOLD = 0.5
CONTROL_VALUES = {
    "control", "controls", "healthy", "health", "normal", "non_depressed",
    "nondepressed", "hc", "cntl", "ctrl", "0", "0.0",
}
MDD_VALUES = {
    "mdd", "major depressive disorder", "majordepressivedisorder", "depressed",
    "depression", "mdd_patient", "case", "cases", "patient", "patients", "1", "1.0",
}
TARGET_ALIASES = {"group", "label", "target", "class", "disease"}
ID_METADATA_NAMES = {"id_ref", "sample_id", "sample", "subject_id", "gsm", "sample_name", "id"}
REQUIRED_ARTIFACTS = (
    "snn_model.keras", "imputer.pkl", "scaler.pkl", "selected_genes.json",
    "label_mapping.json", "feature_metadata.json",
)


def resolve_path(value: str | Path, *, root: Path = ROOT) -> Path:
    """Resolve relative paths against this script's directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}: {exc}") from exc


def validate_artifacts(artifacts_dir: Path) -> tuple[list[str], dict[str, Any], Any, Any]:
    """Load and cross-check artifacts without importing TensorFlow."""
    paths = {name: artifacts_dir / name for name in REQUIRED_ARTIFACTS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing))

    genes_obj = load_json(paths["selected_genes.json"])
    genes = genes_obj.get("selected_genes") if isinstance(genes_obj, dict) else genes_obj
    if not isinstance(genes, list) or not genes or any(not isinstance(g, str) or not g.strip() for g in genes):
        raise ValueError("selected_genes.json must contain a non-empty list of gene names")
    genes = [g.strip() for g in genes]
    if len(set(genes)) != len(genes):
        raise ValueError("selected_genes.json contains duplicate genes")

    mapping = load_json(paths["label_mapping.json"])
    if mapping != {"control": 0, "mdd": 1}:
        raise ValueError("label_mapping.json must be exactly {\"control\": 0, \"mdd\": 1}")
    metadata = load_json(paths["feature_metadata.json"])
    if not isinstance(metadata, dict):
        raise ValueError("feature_metadata.json must contain a JSON object")
    if metadata.get("task") != "mdd_binary_classification":
        raise ValueError("feature_metadata.json task must be 'mdd_binary_classification'")
    if metadata.get("selected_genes") != genes:
        raise ValueError("selected_genes.json and feature_metadata.json gene order/content do not match")
    n_selected = metadata.get("n_selected")
    if isinstance(n_selected, bool) or n_selected != len(genes):
        raise ValueError(f"feature_metadata.json n_selected must equal {len(genes)}")
    if metadata.get("label_mapping") != mapping:
        raise ValueError("feature_metadata.json label_mapping does not match label_mapping.json")
    try:
        imputer = joblib.load(paths["imputer.pkl"])
        scaler = joblib.load(paths["scaler.pkl"])
    except Exception as exc:
        raise ValueError(f"Could not load trusted imputer/scaler pickle artifact: {exc}") from exc
    for name, estimator in (("imputer", imputer), ("scaler", scaler)):
        n_features = getattr(estimator, "n_features_in_", None)
        if n_features is None or int(n_features) != len(genes):
            raise ValueError(f"{name}.n_features_in_ must equal selected gene count ({len(genes)})")
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None and list(names) != genes:
            raise ValueError(f"{name}.feature_names_in_ order/content does not match selected_genes.json")
    return genes, metadata, imputer, scaler


def check_csv_header(path: Path) -> list[str]:
    """Inspect the raw CSV header before pandas can rename duplicate columns."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), None)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read test CSV header from {path}: {exc}") from exc
    if not header or not any(str(name).strip() for name in header):
        raise ValueError(f"Test CSV has no usable header: {path}")
    if len(header) != len(set(header)):
        duplicates = sorted({name for name in header if header.count(name) > 1})
        raise ValueError(f"Test CSV contains duplicate header(s): {duplicates}")
    return header


def choose_target_column(frame: pd.DataFrame, explicit: str | None, metadata: dict[str, Any]) -> str:
    if explicit is not None:
        target = explicit
        if target not in frame.columns:
            raise ValueError(f"Requested target column {target!r} is not present in the test CSV")
        return target
    recorded = metadata.get("target_column")
    if recorded is not None:
        if not isinstance(recorded, str) or recorded not in frame.columns:
            raise ValueError(f"Metadata target_column {recorded!r} is not present in the test CSV; use --target-column if appropriate")
        return recorded
    matches = [col for col in frame.columns if str(col).strip().casefold() in TARGET_ALIASES]
    if len(matches) != 1:
        raise ValueError(f"Could not identify a unique target column from known aliases {sorted(TARGET_ALIASES)}; specify --target-column")
    return matches[0]


def encode_labels(values: pd.Series) -> np.ndarray:
    normalized = values.astype("string").str.strip().str.casefold()
    unknown = sorted(set(normalized.dropna().tolist()) - CONTROL_VALUES - MDD_VALUES)
    if normalized.isna().any() or unknown:
        raise ValueError(
            f"Unknown or missing target label(s): {unknown[:10]}"
            f"{' (and more)' if len(unknown) > 10 else ''}. Expected control synonyms/0 => 0 "
            "or MDD synonyms/1 => 1."
        )
    return normalized.map(lambda value: 0 if value in CONTROL_VALUES else 1).to_numpy(dtype=np.int64)


def validate_model_shapes(model: Any, n_features: int) -> None:
    def single_shape(shape: Any, label: str) -> tuple[Any, ...]:
        if isinstance(shape, list):
            if len(shape) != 1:
                raise ValueError(f"Model must have exactly one {label}; found {len(shape)}")
            shape = shape[0]
        try:
            return tuple(shape)
        except TypeError as exc:
            raise ValueError(f"Could not inspect model {label} shape: {shape!r}") from exc

    in_shape = single_shape(model.input_shape, "input")
    out_shape = single_shape(model.output_shape, "output")
    if len(in_shape) != 2 or in_shape[-1] != n_features:
        raise ValueError(f"Model input shape {in_shape} must be (batch, {n_features})")
    if len(out_shape) != 2 or out_shape[-1] != 1:
        raise ValueError(f"Model output shape {out_shape} must be (batch, 1) for binary probabilities")

def evaluate_predictions(
    y_true: Any,
    probabilities: Any,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    """Calculate test metrics without fitting or training."""

<<<<<<< HEAD
=======
def evaluate_predictions(
    y_true: Any,
    probabilities: Any,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    """Calculate test metrics without fitting or training."""

>>>>>>> 0a5e139fda8138a1a5060108da9cab29b59424de
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError(
            "y_true and probabilities must be non-empty, equal-length 1-D arrays"
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")

    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(
            "probabilities must be finite values in [0, 1]"
        )

    if not np.isfinite(threshold) or threshold != THRESHOLD:
        raise ValueError("Evaluation threshold is fixed at 0.5")

    y = y.astype(np.int64)
    pred = (p >= THRESHOLD).astype(np.int64)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    counts = np.bincount(y, minlength=2)
    two_classes = bool(np.all(counts > 0))

    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "mcc": float(
            matthews_corrcoef(y, pred)
        ),
        "roc_auc": (
            float(roc_auc_score(y, p))
            if two_classes
            else None
        ),
        "roc_auc_reason": (
            None
            if two_classes
            else
            "ROC AUC is undefined because the test set contains only one class."
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, pred))
            if two_classes
            else None
        ),
        "specificity": (
            float(cm[0, 0] / counts[0])
            if counts[0]
            else None
        ),
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_labels": [0, 1],
        "sample_counts": {
            "total": int(len(y)),
            "control_0": int(counts[0]),
            "mdd_1": int(counts[1]),
        },
        "threshold": THRESHOLD,
    }

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved MDD model artifacts on a labeled test CSV; no fitting or training.")
    parser.add_argument("--test", default="mdd_test.csv", help="Labeled MDD test CSV (relative paths resolve against this script).")
<<<<<<< HEAD
=======
    parser.add_argument(
    "--artifacts-dir",
    default="mdd_model_artifacts_small",
    help="Directory containing the saved small MDD SNN artifacts."
)


>>>>>>> 0a5e139fda8138a1a5060108da9cab29b59424de
    parser.add_argument("--output-dir", default="mdd_test_results", help="Directory for JSON metrics and CSV predictions.")
    parser.add_argument(
    "--artifacts-dir",
    default="mdd_model_artifacts_small",
    help="Directory containing the saved small MDD SNN artifacts."
)
    parser.add_argument("--target-column", default=None, help="Target column; otherwise metadata target_column or a unique known alias is used.")
    parser.add_argument("--batch-size", type=int, default=32, help="Prediction batch size (>= 1).")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    test_path = resolve_path(args.test)
    artifacts_dir = resolve_path(args.artifacts_dir)
    output_dir = resolve_path(args.output_dir)
    if not test_path.is_file():
        raise FileNotFoundError(f"Test CSV does not exist: {test_path}")

    genes, metadata, imputer, scaler = validate_artifacts(artifacts_dir)
    training_input = metadata.get("training_input")
    if isinstance(training_input, str) and training_input.strip():
        training_path = resolve_path(training_input)
        if test_path == training_path:
            raise ValueError(f"Refusing to evaluate the recorded training input as a test set: {test_path}")

    header = check_csv_header(test_path)
    try:
        frame = pd.read_csv(test_path)
    except Exception as exc:
        raise ValueError(f"Could not read test CSV {test_path}: {exc}") from exc
    if list(frame.columns) != header:
        raise ValueError("Parsed CSV columns differ from raw header; check malformed CSV quoting or delimiter")
    if frame.empty:
        raise ValueError(f"Test CSV contains no data rows: {test_path}")
    target_col = choose_target_column(frame, args.target_column, metadata)
    y = encode_labels(frame[target_col])

    missing = [gene for gene in genes if gene not in frame.columns]
    if missing:
        raise ValueError(f"Test CSV is missing {len(missing)}/{len(genes)} selected gene(s), e.g. {missing[:10]}")
    raw_genes = frame.loc[:, genes]
    features = raw_genes.apply(pd.to_numeric, errors="coerce")
    newly_missing = raw_genes.notna() & features.isna()
    n_coerced = int(newly_missing.to_numpy().sum())
    if n_coerced:
        warnings.warn(f"Coerced {n_coerced} nonnumeric gene value(s) to missing before imputation.", RuntimeWarning)
    values = features.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Test gene columns contain positive or negative infinity")
    try:
        transformed = scaler.transform(imputer.transform(features))
    except Exception as exc:
        raise ValueError(f"Saved imputer/scaler could not transform selected test genes: {exc}") from exc
    transformed = np.asarray(transformed, dtype=float)
    if transformed.shape != (len(frame), len(genes)):
        raise ValueError(f"Preprocessing returned shape {transformed.shape}; expected {(len(frame), len(genes))}")
    if not np.isfinite(transformed).all():
        raise ValueError("Preprocessed feature matrix contains non-finite values")

    # TensorFlow stays lazy: --help and all file, label, and preprocessing validation work without it.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for inference. Install tensorflow and rerun.") from exc
    model_path = artifacts_dir / "snn_model.keras"
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        raise ValueError(f"Could not load saved Keras model {model_path}: {exc}") from exc
    validate_model_shapes(model, len(genes))
    try:
        probabilities = np.asarray(model.predict(transformed, batch_size=args.batch_size, verbose=0), dtype=float)
    except Exception as exc:
        raise ValueError(f"Model prediction failed: {exc}") from exc
    if probabilities.shape != (len(frame), 1):
        raise ValueError(f"Model returned prediction shape {probabilities.shape}; expected {(len(frame), 1)}")
    probabilities = probabilities[:, 0]
    result = evaluate_predictions(y, probabilities)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.json"
    predictions_path = output_dir / "predictions.csv"
    confusion_path = output_dir / "confusion_matrix.csv"
    result.update({
        "test_csv": str(test_path),
        "artifacts_dir": str(artifacts_dir),
        "target_column": target_col,
        "n_selected": len(genes),
        "coerced_gene_values_to_missing": n_coerced,
    })
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    report = classification_report(
        y, (probabilities >= THRESHOLD).astype(np.int64), labels=[0, 1],
        target_names=["control", "mdd"], output_dict=True, zero_division=0,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    prediction_frame = pd.DataFrame({"original_row_index": frame.index.to_numpy()})
    for col in frame.columns:
        if str(col).strip().casefold() in ID_METADATA_NAMES:
            prediction_frame[col] = frame[col].to_numpy()
    predicted = (probabilities >= THRESHOLD).astype(np.int64)
    prediction_frame["true_label"] = y
    prediction_frame["true_class"] = np.where(y == 0, "control", "mdd")
    prediction_frame["probability_mdd"] = probabilities
    prediction_frame["predicted_label"] = predicted
    prediction_frame["predicted_class"] = np.where(predicted == 0, "control", "mdd")
    prediction_frame.to_csv(predictions_path, index=False)

    cm = np.asarray(result["confusion_matrix"], dtype=int)
    pd.DataFrame(cm, index=["true_control", "true_mdd"], columns=["predicted_control", "predicted_mdd"]).to_csv(confusion_path, index=True, index_label="")

    print("MDD test evaluation metrics (threshold >= 0.5):")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy", "specificity"):
        value = result[key]
        print(f"  {key}: {'null' if value is None else f'{value:.6f}'}")
    if result["roc_auc_reason"]:
        print(f"  roc_auc_reason: {result['roc_auc_reason']}")
    print("Confusion matrix [true rows 0=control,1=mdd; predicted columns 0=control,1=mdd]:")
    print(f"  {cm.tolist()}")
    print(f"Samples: {result['sample_counts']}")
    print(f"Wrote: {metrics_path}\n       {report_path}\n       {predictions_path}\n       {confusion_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
