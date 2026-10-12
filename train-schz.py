#!/usr/bin/env python3
"""Train a schizophrenia-only SELU self-normalizing neural network (SNN).

This script reads training data only. It never loads an independent test set.
Run from the project root, e.g.:
    python train_snn_schizophrenia.py --epochs 80 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent
TARGET_ALIASES = {"group", "label", "target", "class", "disease"}


def resolve_path(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_selected_genes(root: Path) -> tuple[list[str], Path]:
    candidates = [
        root / "pso_smoke_test" / "schz" / "selected_genes.json",
        root / "pso_smoke_test" / "schz" / "selected_genes.csv",
        root / "selected_genes.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".json":
                with path.open(encoding="utf-8") as handle:
                    obj = json.load(handle)
                genes = obj.get("selected_genes") if isinstance(obj, dict) else obj
            else:
                frame = pd.read_csv(path)
                gene_col = next((c for c in frame.columns if str(c).strip().lower() in {"gene", "genes", "feature"}), None)
                if gene_col is None:
                    if frame.shape[1] != 1:
                        raise ValueError("selected_genes.csv must have a gene/feature column or exactly one column")
                    gene_col = frame.columns[0]
                genes = frame[gene_col].dropna().astype(str).tolist()
            if not isinstance(genes, list) or not genes:
                raise ValueError("artifact contains no selected genes")
            genes = [str(g).strip() for g in genes if str(g).strip()]
            if len(genes) != len(set(genes)):
                raise ValueError("selected-gene artifact contains duplicates")
            return genes, path
        except Exception as exc:
            raise ValueError(f"Could not read selected genes from {path}: {exc}") from exc
    raise FileNotFoundError("No schizophrenia selected-gene artifact found (checked PSO JSON/CSV, then root selected_genes.csv).")


def find_target_column(frame: pd.DataFrame) -> str:
    by_lower = {str(col).strip().casefold(): col for col in frame.columns}
    for alias in TARGET_ALIASES:
        if alias in by_lower:
            return by_lower[alias]
    nonnumeric = [col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])]
    if not nonnumeric:
        raise ValueError("Could not identify target: no Group/label/target/class/disease column and no nonnumeric column exists.")
    return nonnumeric[-1]


def encode_labels(values: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    normalized = values.astype("string").str.strip().str.casefold()
    control = {"control", "controls", "healthy", "normal", "0", "0.0"}
    case = {"schizophrenia", "schizophrenic", "scz", "schz", "1", "1.0"}
    unknown = sorted(set(normalized.dropna().tolist()) - control - case)
    if normalized.isna().any() or unknown:
        detail = unknown[:10]
        raise ValueError(f"Unknown or missing target label(s): {detail}{' (and more)' if len(unknown) > 10 else ''}. Expected control=0 or schizophrenia/scz=1.")
    y = normalized.map(lambda x: 0 if x in control else 1).to_numpy(dtype=np.int64)
    return y, {"control": 0, "schizophrenia": 1}


def load_training(path: Path) -> tuple[pd.DataFrame, np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Training input does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Training input must be CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Training input has no rows: {path}")
    target_col = find_target_column(frame)
    y, _ = encode_labels(frame[target_col])
    features = frame.drop(columns=[target_col])
    # Common sample-ID fields are metadata, never model features.
    metadata_names = {"id_ref", "sample_id", "sample", "subject_id"}
    metadata_cols = [c for c in features.columns if str(c).strip().casefold() in metadata_names]
    if metadata_cols:
        features = features.drop(columns=metadata_cols)
    return features, y, str(target_col)


def validate_features(frame: pd.DataFrame, genes: list[str], source: Path) -> pd.DataFrame:
    missing = [gene for gene in genes if gene not in frame.columns]
    if missing:
        raise ValueError(
            f"Selected schizophrenia genes from the artifact do not overlap sufficiently with {source}. "
            f"Missing {len(missing)}/{len(genes)} gene(s), e.g. {missing[:10]}."
        )
    selected = frame.loc[:, genes].apply(pd.to_numeric, errors="coerce")
    if selected.isna().all(axis=0).any():
        bad = selected.columns[selected.isna().all(axis=0)].tolist()
        raise ValueError(f"Selected gene columns contain no numeric values: {bad}")
    return selected


def is_preprocessed_pso(source: Path, features: pd.DataFrame, genes: list[str], root: Path) -> bool:
    """Conservatively detect the known PSO training matrix (metadata + exact feature set)."""
    expected_dir = root / "pso_smoke_test" / "schz"
    if source.resolve().parent != expected_dir.resolve() or source.name.casefold() != "snn_train_input.csv":
        return False
    if list(features.columns) != genes:
        return False
    metadata_path = expected_dir / "feature_metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        declared = metadata.get("selected_genes", [])
        if declared != genes:
            return False
    except (OSError, json.JSONDecodeError):
        return False
    values = features.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        return False
    # Metadata, canonical PSO filename, exact feature order, and train-standardized shape.
    means = values.mean(axis=0).abs().to_numpy()
    stds = values.std(axis=0, ddof=1).to_numpy()
    return bool(np.all(means < 0.08) and np.all(np.abs(stds - 1.0) < 0.08))


def build_snn_model(input_dim: int, seed: int, tf: Any):
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="selu", kernel_initializer="lecun_normal"),
        layers.AlphaDropout(0.1),
        layers.Dense(32, activation="selu", kernel_initializer="lecun_normal"),
        layers.AlphaDropout(0.1),
        layers.Dense(16, activation="selu", kernel_initializer="lecun_normal"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def preprocess_fit(train_x: pd.DataFrame, other_x: pd.DataFrame | None = None):
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_x)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    if other_x is None:
        return train_scaled, imputer, scaler
    return train_scaled, scaler.transform(imputer.transform(other_x)), imputer, scaler


def run_cv(raw_features: pd.DataFrame, y: np.ndarray, args: argparse.Namespace, tf: Any) -> dict[str, list[float]]:
    counts = np.bincount(y, minlength=2)
    if counts.min() < 5:
        raise ValueError(f"5-fold StratifiedKFold needs at least 5 samples per class; found counts={counts.tolist()}.")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    metrics: dict[str, list[float]] = {key: [] for key in ("accuracy", "precision", "recall", "f1", "auc")}
    print("\n5-fold stratified CV (preprocessing fit separately on each fold's training rows):")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(raw_features, y), 1):
        x_train_raw = raw_features.iloc[train_idx]
        x_val_raw = raw_features.iloc[val_idx]
        x_train, x_val, _, _ = preprocess_fit(x_train_raw, x_val_raw)
        y_train, y_val = y[train_idx], y[val_idx]
        model = build_snn_model(len(raw_features.columns), args.seed + fold - 1, tf)
        classes = np.array([0, 1], dtype=np.int64)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0, class_weight=class_weight)
        probability = model.predict(x_val, verbose=0).ravel()
        prediction = (probability >= 0.5).astype(np.int64)
        values = {
            "accuracy": accuracy_score(y_val, prediction),
            "precision": precision_score(y_val, prediction, zero_division=0),
            "recall": recall_score(y_val, prediction, zero_division=0),
            "f1": f1_score(y_val, prediction, zero_division=0),
            "auc": roc_auc_score(y_val, probability),
        }
        for key, value in values.items():
            metrics[key].append(float(value))
        print("Fold {}: ".format(fold) + " | ".join(f"{k.upper()}={v:.4f}" for k, v in values.items()))
        tf.keras.backend.clear_session()
    print("CV mean ± std: " + " | ".join(f"{k.upper()}={np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()))
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a schizophrenia-only SNN using training data only.")
    parser.add_argument("--train", default="pso_smoke_test/schz/snn_train_input.csv", help="Training CSV; defaults to the available PSO training matrix, otherwise schz_train.csv.")
    parser.add_argument("--artifacts-dir", default="schizophrenia_artifacts", help="Output directory for the trained model and preprocessing metadata.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive integers.")
    np.random.seed(args.seed)
    genes, genes_path = read_selected_genes(ROOT)
    requested_train = resolve_path(args.train)
    if args.train == "pso_smoke_test/schz/snn_train_input.csv" and not requested_train.is_file():
        requested_train = resolve_path("schz_train.csv")
    features_all, y, target_col = load_training(requested_train)
    selected = validate_features(features_all, genes, requested_train)
    preprocessed_pso = is_preprocessed_pso(requested_train, features_all, genes, ROOT)

    # CV always uses raw training measurements where available, so every fold fits
    # its own imputer/scaler. The final model uses the prioritized PSO input as-is
    # when its metadata and standardized feature shape prove it is already processed.
    if preprocessed_pso:
        raw_path = resolve_path("schz_train.csv")
        if not raw_path.is_file():
            raise FileNotFoundError("PSO input is already globally preprocessed, so leakage-safe fold CV requires the raw schizophrenia training file schz_train.csv.")
        raw_features_all, raw_y, _ = load_training(raw_path)
        if len(raw_y) != len(y) or not np.array_equal(raw_y, y):
            raise ValueError("Raw schz_train.csv labels/order do not match the PSO training matrix; cannot guarantee leakage-safe CV alignment.")
        cv_features = validate_features(raw_features_all, genes, raw_path)
    else:
        cv_features = selected
    print(f"Training input: {requested_train}")
    print(f"Target column: {target_col} | samples: {len(y)} | selected genes: {len(genes)}")
    print(f"Selected genes artifact: {genes_path}")
    if preprocessed_pso:
        print("Detected standardized PSO training input; final model will not preprocess it again.")
    else:
        print("Raw training input detected; imputation/scaling will be fit on each CV train fold and then on all training rows.")

    # TensorFlow is imported after argument parsing, allowing --help even if it is absent.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to train this model. Install tensorflow and rerun.") from exc
    tf.keras.utils.set_random_seed(args.seed)
    run_cv(cv_features, y, args, tf)

    artifacts_dir = resolve_path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if preprocessed_pso:
        pso_dir = requested_train.parent
        imputer_path, scaler_path = pso_dir / "imputer.pkl", pso_dir / "scaler.pkl"
        if not imputer_path.is_file() or not scaler_path.is_file():
            raise FileNotFoundError("PSO matrix is marked preprocessed but its imputer.pkl/scaler.pkl artifacts are missing.")
        imputer, scaler = joblib.load(imputer_path), joblib.load(scaler_path)
        final_x = selected.to_numpy(dtype=np.float64)
    else:
        final_x, imputer, scaler = preprocess_fit(selected)
    counts = np.bincount(y, minlength=2)
    if counts.min() == 0:
        raise ValueError(f"Final training requires both classes; found counts={counts.tolist()}.")
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    final_model = build_snn_model(len(genes), args.seed, tf)
    final_model.fit(final_x, y, epochs=args.epochs, batch_size=args.batch_size, verbose=1, class_weight=class_weight)

    final_model.save(artifacts_dir / "snn_model.keras")
    joblib.dump(imputer, artifacts_dir / "imputer.pkl")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    (artifacts_dir / "selected_genes.json").write_text(json.dumps(genes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "task": "schizophrenia_binary_classification", "training_input": str(requested_train),
        "target_column": target_col, "n_train_samples": int(len(y)), "n_selected": len(genes),
        "selected_genes": genes, "preprocessed_input_used_as_is": bool(preprocessed_pso),
        "preprocessing": "PSO preprocessed input (no second transform)" if preprocessed_pso else "SimpleImputer(strategy='median') then StandardScaler",
        "cv": {"strategy": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": args.seed,
               "fold_preprocessing_fit_on_fold_train_only": True},
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
    }
    (artifacts_dir / "feature_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (artifacts_dir / "label_mapping.json").write_text(json.dumps({"control": 0, "schizophrenia": 1}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nTraining complete. Artifacts saved to: {artifacts_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
