import argparse
import json
import os
from pathlib import Path

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
)
import tensorflow as tf



def parse_args():
    parser = argparse.ArgumentParser(description="ارزیابی مدل MDD روی داده تست")
    parser.add_argument(
        "--test-csv",
        type=str,
        default="mdd_test.csv",
        help="مسیر فایل تست CSV",

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
>>>>>>> a3691b9f64bcce215e4b01a4b1606a1548d6071c
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="mdd_model_artifacts",
        help="پوشه حاوی مدل و ابزارهای پیش‌پردازش",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mdd_test_results",
        help="مسیر ذخیره نتایج تست",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7216,
        help="آستانه تصمیم‌گیری برای کلاس MDD (پیش‌فرض: 0.7216)",
    )
    return parser.parse_args()


def load_artifacts(artifacts_dir: Path):
    print(f"بارگذاری تنظیمات و ابزارهای پیش‌پردازش از: {artifacts_dir}")

    # ۱. مدل
    model_path = artifacts_dir / "snn_model.keras"
    if not model_path.exists():
        model_path = artifacts_dir / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"فایل مدل در {artifacts_dir} یافت نشد.")
    model = tf.keras.models.load_model(model_path)

    # ۲. اسکیلر و Imputer
    scaler_path = artifacts_dir / "scaler.pkl"
    imputer_path = artifacts_dir / "imputer.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    imputer = joblib.load(imputer_path) if imputer_path.exists() else None

    # ۳. لیست ژن‌ها
    genes_path = artifacts_dir / "selected_genes.json"
    meta_path = artifacts_dir / "feature_metadata.json"
    if genes_path.exists():
        with open(genes_path, "r", encoding="utf-8") as f:
            feature_genes = json.load(f)
    elif meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            feature_genes = meta.get("feature_names", meta.get("selected_genes", []))
    else:
        raise FileNotFoundError("لیست ژن‌های انتخابی یافت نشد.")

    # ۴. مپینگ برچسب‌ها
    mapping_path = artifacts_dir / "label_mapping.json"
    encoder_path = artifacts_dir / "label_encoder.pkl"
    label_map = None
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    elif encoder_path.exists():
        encoder = joblib.load(encoder_path)
        label_map = {cls: idx for idx, cls in enumerate(encoder.classes_)}

    return model, scaler, imputer, feature_genes, label_map


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # بارگذاری فایل تست
    test_csv = Path(args.test_csv)
    if not test_csv.exists():
        raise FileNotFoundError(f"فایل داده تست یافت نشد: {test_csv}")
    df_test = pd.read_csv(test_csv)

    model, scaler, imputer, feature_genes, label_map = load_artifacts(artifacts_dir)

    # جداسازی لیبل
        # جداسازی لیبل
    target_col = None
    for col in ["target", "label", "condition", "group", "Group", "status", "diagnosis", "class"]:
        if col in df_test.columns:
            target_col = col
            break


    if target_col is None:
        raise ValueError("ستون برچسب (مانند target یا label) در فایل تست پیدا نشد.")

    y_raw = df_test[target_col]
    
    # نگاشت برچسب به 0 و 1
    if label_map is not None:
        y_true = y_raw.map(label_map).values
    else:
        # نگاشت پیش‌فرض: control -> 0 و بقیه -> 1
        y_true = (
            y_raw.astype(str)
            .str.lower()
            .apply(lambda x: 0 if "control" in x or x == "0" else 1)
            .values
        )

    # انتخاب ویژگی‌ها
    available_genes = [g for g in feature_genes if g in df_test.columns]
    X_test = df_test[available_genes].copy()

    # پر کردن ستون‌های مفقود اگر وجود داشتند
    for missing_gene in set(feature_genes) - set(available_genes):
        X_test[missing_gene] = 0.0
    X_test = X_test[feature_genes]

    # اعمال Imputer و Scaler
    if imputer is not None:
        X_test = imputer.transform(X_test)
    if scaler is not None:
        X_test = scaler.transform(X_test)

    # پیش‌بینی احتمال
    probs = model.predict(X_test)
    if probs.ndim > 1 and probs.shape[1] > 1:
        probs = probs[:, 1]
    else:
        probs = probs.ravel()

    # اعمال آستانه تصمیم‌گیری
    preds = (probs >= args.threshold).astype(int)

    # محاسبه معیارها
    acc = accuracy_score(y_true, preds)
    bacc = balanced_accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 45)
    print(f"نتایج ارزیابی تست با آستانه {args.threshold}:")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bacc:.4f}")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"  Specificity:       {spec:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  ROC-AUC:           {auc:.4f}")
    print("\nماتریس درهم‌ریختگی (Confusion Matrix):")
    print(cm)
    print("=" * 45)

    # ذخیره پیش‌بینی‌ها
    results_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_prob": probs,
            "predicted_label": preds,
        }
    )
    save_path = output_dir / "predictions.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nفایل نتایج در '{save_path}' ذخیره شد.")


if __name__ == "__main__":
    main()
