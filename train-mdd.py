#!/usr/bin/env python3
"""Train an MDD-only SELU self-normalizing neural network (SNN).

WARNING: selected genes may have been selected earlier using the FULL training
set. CV is therefore conditional on that fixed gene list and can be optimistic
for the complete feature-selection pipeline; this is NOT nested CV. This script
uses training data only and never loads an independent test set.

Run from the project root (relative paths resolve against this script):
    python train_mdd_small_snn.py --train mdd_train.csv \\
        --genes mdd_artifacts/selected_genes.json --epochs 80 --batch-size 8
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent
TARGET_ALIASES = {"group", "label", "target", "class", "disease"}
ID_METADATA_NAMES = {"id_ref", "sample_id", "sample", "subject_id", "gsm", "sample_name", "id"}
CONTROL_VALUES = {
    "control", "controls", "healthy", "health", "normal", "non_depressed",
    "nondepressed", "hc", "cntl", "ctrl", "0", "0.0",
}
MDD_VALUES = {
    "mdd", "major depressive disorder", "majordepressivedisorder", "depressed",
    "depression", "mdd_patient", "case", "cases", "patient", "patients", "1", "1.0",
}


def resolve_path(value: str | Path, *, root: Path = ROOT) -> Path:
    """Resolve relative CLI paths against the script directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_selected_genes(path: Path) -> list[str]:
    """Read genes from a JSON list/object or CSV gene/genes/feature column."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Selected-genes artifact not found: {path}. Pass the MDD artifact explicitly with --genes."
        )
    try:
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as handle:
                obj = json.load(handle)
            genes = obj.get("selected_genes") if isinstance(obj, dict) else obj
        else:
            frame = pd.read_csv(path)
            gene_col = next(
                (c for c in frame.columns if str(c).strip().casefold() in {"gene", "genes", "feature"}),
                None,
            )
            if gene_col is None:
                if frame.shape[1] != 1:
                    raise ValueError("CSV artifact must have a gene/genes/feature column or exactly one column")
                gene_col = frame.columns[0]
            genes = frame[gene_col].dropna().astype(str).tolist()
        if not isinstance(genes, list) or not genes:
            raise ValueError("artifact contains no selected genes")
        genes = [str(g).strip() for g in genes if str(g).strip()]
        if not genes:
            raise ValueError("artifact contains no selected genes")
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
        raise ValueError(
            "Could not identify target: no Group/label/target/class/disease column and no nonnumeric column exists."
        )
    return nonnumeric[-1]


def encode_labels(values: pd.Series) -> np.ndarray:
    """Fixed encoding: controls => 0 and MDD cases => 1; reject unknown labels."""
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
    metadata_cols = [c for c in features.columns if str(c).strip().casefold() in ID_METADATA_NAMES]
    if metadata_cols:
        features = features.drop(columns=metadata_cols)
    return features, y, str(target_col)


def validate_features(frame: pd.DataFrame, genes: list[str], source: Path) -> pd.DataFrame:
    """All genes must exist and be numeric-coercible; preserve artifact order."""
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


def build_snn_model(input_dim: int, seed: int, tf: Any, hidden_layers: list[int]):
    """Build configurable SELU network; hidden layers use LeCunNormal."""
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    model_layers = [layers.Input(shape=(input_dim,))]
    for index, units in enumerate(hidden_layers):
        model_layers.append(
            layers.Dense(units, activation="selu", kernel_initializer="lecun_normal", name=f"hidden_{index + 1}")
        )
        if index < len(hidden_layers) - 1:
            model_layers.append(layers.AlphaDropout(0.1, name=f"alpha_dropout_{index + 1}"))
    model_layers.append(layers.Dense(1, activation="sigmoid", name="output"))
    model = tf.keras.Sequential(model_layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
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


def _metric_values(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "specificity": specificity,
        "mcc": float(matthews_corrcoef(y_true, prediction)),
    }


def run_cv(
    raw_features: pd.DataFrame, y: np.ndarray, args: argparse.Namespace, tf: Any
) -> tuple[dict[str, list[float]], list[int]]:
    counts = np.bincount(y, minlength=2)
    if counts.min() < args.cv_folds:
        raise ValueError(
            f"{args.cv_folds}-fold StratifiedKFold needs at least {args.cv_folds} samples per class; "
            f"found counts={counts.tolist()}."
        )
    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    metric_names = (
        "accuracy", "precision", "recall", "f1", "auc", "balanced_accuracy", "specificity", "mcc"
    )
    metrics: dict[str, list[float]] = {key: [] for key in metric_names}
    best_epochs: list[int] = []
    print(f"\n{args.cv_folds}-fold stratified CV (fold-local preprocessing and inner early-stopping validation):")
    print("NOTE: CV is conditional on the fixed pre-selected gene list; it may be optimistic for the full feature-selection pipeline.")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(raw_features, y), 1):
        x_outer_train = raw_features.iloc[train_idx]
        y_outer_train, y_outer_val = y[train_idx], y[val_idx]
        inner_counts = np.bincount(y_outer_train, minlength=2)
        if inner_counts.min() < 2:
            raise ValueError(f"Fold {fold}: inner stratified split needs at least 2 samples per class; found {inner_counts.tolist()}.")
        inner_split = StratifiedShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed + fold - 1)
        inner_train_rel, inner_val_rel = next(inner_split.split(x_outer_train, y_outer_train))
        x_inner_train_raw = x_outer_train.iloc[inner_train_rel]
        x_inner_val_raw = x_outer_train.iloc[inner_val_rel]
        y_inner_train, y_inner_val = y_outer_train[inner_train_rel], y_outer_train[inner_val_rel]
        inner_counts = np.bincount(y_inner_train, minlength=2)
        if inner_counts.min() == 0 or np.bincount(y_inner_val, minlength=2).min() == 0:
            raise ValueError(
                f"Fold {fold}: validation_fraction={args.validation_fraction} leaves a class absent from an inner partition."
            )

        x_inner_train, x_inner_val, inner_imputer, inner_scaler = preprocess_fit(x_inner_train_raw, x_inner_val_raw)
        # Transform outer validation with the inner-training-only preprocessing; it is never used for stopping.
        x_outer_val = inner_scaler.transform(inner_imputer.transform(raw_features.iloc[val_idx]))
        model = build_snn_model(len(raw_features.columns), args.seed + fold - 1, tf, args.hidden_layers)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta, mode="min", restore_best_weights=True,
        )
        history = model.fit(
            x_inner_train, y_inner_train,
            validation_data=(x_inner_val, y_inner_val),
            epochs=args.epochs, batch_size=args.batch_size, verbose=0,
            class_weight=class_weights(y_inner_train), callbacks=[callback],
        )
        val_losses = history.history.get("val_loss", [])
        best_epoch = int(np.argmin(val_losses) + 1) if val_losses else int(len(history.epoch))
        best_epochs.append(best_epoch)
        probability = model.predict(x_outer_val, verbose=0).ravel()
        prediction = (probability >= 0.5).astype(np.int64)
        values = _metric_values(y_outer_val, prediction, probability)
        for key, value in values.items():
            metrics[key].append(float(value))
        print(f"Fold {fold} (best epoch={best_epoch}): " + " | ".join(f"{k.upper()}={v:.4f}" for k, v in values.items()))
        del inner_imputer, inner_scaler
        tf.keras.backend.clear_session()
    print("Best epochs by fold: " + ", ".join(map(str, best_epochs)))
    print("CV mean ± std: " + " | ".join(f"{k.upper()}={np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()))
    return metrics, best_epochs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MDD-only SELU SNN using training data only (no test set).")
    parser.add_argument("--train", default="mdd_train.csv", help="MDD training CSV; relative paths resolve against this script.")
    parser.add_argument("--genes", default="mdd_artifacts/selected_genes.json", help="Selected-gene JSON/CSV artifact.")
    parser.add_argument("--artifacts-dir", default="mdd_model_artifacts", help="Output directory for model artifacts.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum epochs for each CV fold.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[32, 16], metavar="UNITS",
                        help="Positive hidden layer widths (default: 32 16).")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="EarlyStopping patience for each inner validation split (default: 10).")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="Minimum val_loss improvement to count (default: 0).")
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        help="Inner stratified validation fraction within each outer training fold (0,1).")
    parser.add_argument("--final-epochs", type=int, default=None,
                        help="Override final-model epochs; default is rounded median best CV epoch.")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.cv_folds < 2:
        raise ValueError("--epochs and --batch-size must be >= 1 and --cv-folds must be >= 2.")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1.")
    if not np.isfinite(args.early_stopping_min_delta) or args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be finite and >= 0.")
    if not np.isfinite(args.validation_fraction) or not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be strictly between 0 and 1.")
    if args.final_epochs is not None and args.final_epochs < 1:
        raise ValueError("--final-epochs must be >= 1 when provided.")
    if not args.hidden_layers or any(units < 1 for units in args.hidden_layers):
        raise ValueError("--hidden-layers must contain one or more positive integers.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
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
    print("WARNING: genes were selected earlier using the full training set, so CV is conditional on a fixed gene list and may be optimistic. The independent test set remains untouched.")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to train this model. Install tensorflow and rerun.") from exc
    tf.keras.utils.set_random_seed(args.seed)

    cv_metrics, best_epochs = run_cv(selected, y, args, tf)
    artifacts_dir = resolve_path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if counts.min() == 0:
        raise ValueError(f"Final training requires both classes; found counts={counts.tolist()}.")
    final_epochs = args.final_epochs
    if final_epochs is None:
        median_best = float(np.median(best_epochs))
        final_epochs = max(1, int(np.floor(median_best + 0.5)))
    final_x, imputer, scaler = preprocess_fit(selected)
    final_model = build_snn_model(len(genes), args.seed, tf, args.hidden_layers)
    final_model.fit(
        final_x, y, epochs=final_epochs, batch_size=args.batch_size, verbose=1,
        class_weight=class_weights(y),
    )

    label_mapping = {"control": 0, "mdd": 1}
    final_model.save(artifacts_dir / "snn_model.keras")
    joblib.dump(imputer, artifacts_dir / "imputer.pkl")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    (artifacts_dir / "selected_genes.json").write_text(
        json.dumps(genes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (artifacts_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
        "decision_threshold": 0.5,
        "preprocessing": "SimpleImputer(strategy='median') then StandardScaler (fit on all training rows for final model)",
        "cv": {
            "strategy": "StratifiedKFold",
            "n_splits": args.cv_folds,
            "shuffle": True,
            "random_state": args.seed,
            "fold_preprocessing_fit_on_fold_train_only": True,
            "early_stopping_validation": "StratifiedShuffleSplit drawn only from each outer fold's training rows; outer validation fold used only for evaluation",
            "early_stopping": {
                "monitor": "val_loss",
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
                "restore_best_weights": True,
                "validation_fraction": args.validation_fraction,
                "max_epochs": args.epochs,
            },
            "best_epochs": best_epochs,
            "note": "Conditional on a fixed pre-selected gene list; optimistic for full feature selection. NOT nested CV. Independent test set untouched.",
            "fold_metrics": cv_metrics,
            "mean": {k: float(np.mean(v)) for k, v in cv_metrics.items()},
            "std": {k: float(np.std(v)) for k, v in cv_metrics.items()},
        },
        "model": {
            "architecture": "SELU self-normalizing neural network",
            "layers": args.hidden_layers,
            "activation": "selu",
            "kernel_initializer": "LeCunNormal",
            "dropout": "AlphaDropout(0.1) between hidden layers",
            "output": "sigmoid",
            "optimizer": "Adam",
            "learning_rate": 0.001,
        },
        "seed": args.seed,
        "max_epochs": args.epochs,
        "final_epochs": int(final_epochs),
        "final_epochs_source": "--final-epochs" if args.final_epochs is not None else "rounded median of outer-fold best epochs",
        "batch_size": args.batch_size,
    }
    (artifacts_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nFinal model epochs: {final_epochs} (fold best epochs: {best_epochs})")
    print(f"Training complete. Artifacts saved to: {artifacts_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
