#!/usr/bin/env python3
"""Train an MDD-only SELU self-normalizing neural network (SNN).

WARNING (read carefully): the selected genes in this pipeline were selected
EARLIER using the FULL training set. The ordinary cross-validation performed
here is therefore conditional on that fixed gene list, which makes it an
OPTIMISTIC estimate of the complete feature-selection + training pipeline.
This is NOT nested CV. The independent test set must remain untouched until
final evaluation of the saved model.

This script reads MDD training data only. It never loads an independent test
set. Run from the project root, e.g.:
    python train_mdd.py --train mdd_train.csv --genes mdd_artifacts/selected_genes.json --epochs 80 --batch-size 8
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
ID_METADATA_NAMES = {"id_ref", "sample_id", "sample", "subject_id", "gsm", "sample_name", "id"}
CONTROL_VALUES = {"control", "controls", "healthy", "health", "normal", "non_depressed",
                  "nondepressed", "hc", "cntl", "ctrl", "0", "0.0"}
MDD_VALUES = {"mdd", "major depressive disorder", "majordepressivedisorder",
              "depressed", "depression", "mdd_patient", "case", "cases",
              "patient", "patients", "1", "1.0"}


def resolve_path(value: str | Path, *, root: Path = ROOT) -> Path:
    """Resolve relative CLI paths against the script directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_selected_genes(path: Path) -> list[str]:
    """Read selected genes from JSON (list or {"selected_genes": [...]}) or CSV
    (gene/genes/feature column or a one-column file). No generic fallbacks."""
    if not path.is_file():
        raise FileNotFoundError(f"Selected-genes artifact not found: {path}. "
                                "Pass the MDD artifact explicitly with --genes.")
    try:
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as handle:
                obj = json.load(handle)
            genes = obj.get("selected_genes") if isinstance(obj, dict) else obj
        else:
            frame = pd.read_csv(path)
            gene_col = next((c for c in frame.columns if str(c).strip().casefold() in {"gene", "genes", "feature"}), None)
            if gene_col is None:
                if frame.shape[1] != 1:
                    raise ValueError("CSV artifact must have a gene/genes/feature column or exactly one column")
                gene_col = frame.columns[0]
            genes = frame[gene_col].dropna().astype(str).tolist()
        if not isinstance(genes, list) or not genes:
            raise ValueError("artifact contains no selected genes")
        genes = [str(g).strip() for g in genes if str(g).strip()]
        if len(genes) != len(set(genes)):
            raise ValueError("selected-gene artifact contains duplicates")
        return genes
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read selected genes from {path}: {exc}") from exc


def find_target_column(frame: pd.DataFrame) -> str:
    by_lower = {str(col).strip().casefold(): col for col in frame.columns}
    for alias in TARGET_ALIASES:
        if alias in by_lower:
            return by_lower[alias]
    nonnumeric = [col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])]
    if not nonnumeric:
        raise ValueError("Could not identify target: no Group/label/target/class/disease column and no nonnumeric column exists.")
    return nonnumeric[-1]


def encode_labels(values: pd.Series) -> np.ndarray:
    """Fixed MDD encoding: control/CNTL/healthy/numeric 0 => 0; MDD/CASE/patient/
    numeric 1 => 1. Missing or unrecognized values are rejected."""
    normalized = values.astype("string").str.strip().str.casefold()
    unknown = sorted(set(normalized.dropna().tolist()) - CONTROL_VALUES - MDD_VALUES)
    if normalized.isna().any() or unknown:
        detail = unknown[:10]
        raise ValueError(
            f"Unknown or missing target label(s): {detail}{' (and more)' if len(unknown) > 10 else ''}. "
            "Expected control/control synonyms/CNTL/0 => 0 or mdd/major depressive disorder synonyms/CASE/1 => 1."
        )
    return normalized.map(lambda x: 0 if x in CONTROL_VALUES else 1).to_numpy(dtype=np.int64)


def load_training(path: Path) -> tuple[pd.DataFrame, np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Training input does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Training input must be CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Training input has no rows: {path}")
    target_col = find_target_column(frame)
    y = encode_labels(frame[target_col])
    features = frame.drop(columns=[target_col])
    # Common sample-ID fields are metadata, never model features.
    metadata_cols = [c for c in features.columns if str(c).strip().casefold() in ID_METADATA_NAMES]
    if metadata_cols:
        features = features.drop(columns=metadata_cols)
    return features, y, str(target_col)


def validate_features(frame: pd.DataFrame, genes: list[str], source: Path) -> pd.DataFrame:
    """All genes must exist and be numeric-coercible; artifact order is kept."""
    missing = [gene for gene in genes if gene not in frame.columns]
    if missing:
        raise ValueError(
            f"Selected MDD genes from the artifact do not exist in {source}. "
            f"Missing {len(missing)}/{len(genes)} gene(s), e.g. {missing[:10]}."
        )
    selected = frame.loc[:, genes].apply(pd.to_numeric, errors="coerce")
    if selected.isna().all(axis=0).any():
        bad = selected.columns[selected.isna().all(axis=0)].tolist()
        raise ValueError(f"Selected gene columns contain no numeric values: {bad}")
    return selected


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


def class_weights(y: np.ndarray) -> dict[int, float]:
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1], dtype=np.int64), y=y)
    return {int(c): float(w) for c, w in zip([0, 1], weights)}


def run_cv(raw_features: pd.DataFrame, y: np.ndarray, args: argparse.Namespace, tf: Any) -> dict[str, list[float]]:
    counts = np.bincount(y, minlength=2)
    if counts.min() < args.cv_folds:
        raise ValueError(f"{args.cv_folds}-fold StratifiedKFold needs at least {args.cv_folds} samples per class; found counts={counts.tolist()}.")
    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    metrics: dict[str, list[float]] = {key: [] for key in ("accuracy", "precision", "recall", "f1", "auc")}
    print(f"\n{args.cv_folds}-fold stratified CV (imputer/scaler fit separately on each fold's training rows only):")
    print("NOTE: CV is conditional on the fixed pre-selected gene list; it may be optimistic for the full feature-selection pipeline.")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(raw_features, y), 1):
        x_train_raw = raw_features.iloc[train_idx]
        x_val_raw = raw_features.iloc[val_idx]
        x_train, x_val, _, _ = preprocess_fit(x_train_raw, x_val_raw)
        y_train, y_val = y[train_idx], y[val_idx]
        model = build_snn_model(len(raw_features.columns), args.seed + fold - 1, tf)
        model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0, class_weight=class_weights(y_train))
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
    parser = argparse.ArgumentParser(description="Train an MDD-only SELU SNN using MDD training data only (no test set).")
    parser.add_argument("--train", default="mdd_train.csv", help="MDD training CSV (relative paths resolve against this script's directory).")
    parser.add_argument("--genes", default="mdd_artifacts/selected_genes.json", help="Selected-genes artifact for MDD: JSON list, {'selected_genes': [...]}, or CSV with gene/genes/feature column (or one column).")
    parser.add_argument("--artifacts-dir", default="mdd_model_artifacts", help="Output directory (kept separate from PSO output so PSO artifacts are not overwritten).")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.cv_folds < 2:
        raise ValueError("--epochs and --batch-size must be >= 1 and --cv-folds >= 2.")

    np.random.seed(args.seed)
    train_path = resolve_path(args.train)
    genes_path = resolve_path(args.genes)
    genes = read_selected_genes(genes_path)

    features_all, y, target_col = load_training(train_path)
    selected = validate_features(features_all, genes, train_path)

    print(f"Training input: {train_path}")
    print(f"Target column: {target_col} | samples: {len(y)} | selected genes: {len(genes)}")
    print(f"Selected genes artifact: {genes_path}")
    counts = np.bincount(y, minlength=2)
    print(f"Class counts: control(0)={counts[0]} mdd(1)={counts[1]}")
    print("WARNING: genes were selected earlier using the full training set, so the CV below is conditional on a fixed gene list and can be optimistic for the full feature-selection pipeline. The independent test set remains untouched until final evaluation.")

    # TensorFlow is imported after argument parsing, allowing --help even if it is absent.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to train this model. Install tensorflow and rerun.") from exc
    tf.keras.utils.set_random_seed(args.seed)

    cv_metrics = run_cv(selected, y, args, tf)

    artifacts_dir = resolve_path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Final preprocessing and model on ALL training rows.
    final_x, imputer, scaler = preprocess_fit(selected)
    if counts.min() == 0:
        raise ValueError(f"Final training requires both classes; found counts={counts.tolist()}.")
    final_model = build_snn_model(len(genes), args.seed, tf)
    final_model.fit(final_x, y, epochs=args.epochs, batch_size=args.batch_size, verbose=1, class_weight=class_weights(y))

    label_mapping = {"control": 0, "mdd": 1}
    final_model.save(artifacts_dir / "snn_model.keras")
    joblib.dump(imputer, artifacts_dir / "imputer.pkl")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    (artifacts_dir / "selected_genes.json").write_text(json.dumps(genes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (artifacts_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "task": "mdd_binary_classification",
        "training_input": str(train_path),
        "genes_artifact": str(genes_path),
        "target_column": target_col,
        "n_train_samples": int(len(y)),
        "class_counts": {"control_0": int(counts[0]), "mdd_1": int(counts[1])},
        "n_selected": len(genes),
        "selected_genes": genes,
        "label_mapping": label_mapping,
        "preprocessing": "SimpleImputer(strategy='median') then StandardScaler (fit on all training rows for final model)",
        "cv": {
            "strategy": "StratifiedKFold",
            "n_splits": args.cv_folds,
            "shuffle": True,
            "random_state": args.seed,
            "fold_preprocessing_fit_on_fold_train_only": True,
            "note": "Conditional on a fixed pre-selected gene list; optimistic for the full feature-selection pipeline. NOT nested CV. Independent test set untouched.",
            "fold_metrics": cv_metrics,
            "mean": {k: float(np.mean(v)) for k, v in cv_metrics.items()},
            "std": {k: float(np.std(v)) for k, v in cv_metrics.items()},
        },
        "model": {"layers": [64, 32, 16], "activation": "selu", "dropout": "AlphaDropout(0.1)", "output": "sigmoid"},
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    (artifacts_dir / "feature_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nTraining complete. Artifacts saved to: {artifacts_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
