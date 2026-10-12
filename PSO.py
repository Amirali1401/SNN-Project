#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSO_schizophrenia_train_only.py — Schizophrenia feature selection on the
TRAINING SET ONLY.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  THIS SCRIPT MUST BE RUN ON THE **SCHIZOPHRENIA TRAIN CSV ONLY**         !
!  (e.g. schz_train.csv). The training CSV must contain a 'Group' label    !
!  column ('control'/'schizophrenia') plus numeric feature columns.        !
!  Metadata columns sample_id / ID_REF / Group / genotype / treatment are  !
!  ignored (ID_REF is ignored case-insensitively).                         !
!
!  The FINAL INDEPENDENT TEST SET (schz_test.csv) must NEVER be passed to  !
!  this script. Evaluation on the test set is done in a separate step that !
!  ONLY LOADS the saved artifacts (selected_genes.json, imputer.pkl,       !
!  feature_metadata.json, selected_genes.csv, pso_results.csv,             !
!  snn_train_input.csv). The test script must NEVER rerun PSO / feature    !
!  selection.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Design notes:
- Deterministic "MOPSO-like" search: instead of a stochastic particle swarm,
  we use a fully deterministic wrapper search (RandomForest importance
  ranking with stability tie-breaking). It always selects EXACTLY
  --target-genes (default 38) features and gives a stable feature order
  across runs with the same --seed.
- Cross-validation is STRATIFIED and performed ONLY inside the training data.
- The median imputer and StandardScaler are fitted ONLY on the training data
  (train-only fit).

Example:
    python PSO_schizophrenia_train_only.py --input schz_train.csv \
        --output-dir schizophrenia_artifacts --target-genes 38 --seed 42 \
        --particles 30 --iterations 40 --cv-folds 5 --estimators 200

Only standard library + numpy, pandas, scikit-learn, joblib are used.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

META_COLUMNS = {"sample_id", "id_ref", "group", "genotype", "treatment"}
LABEL_ALIASES = {
    "control": "control", "controls": "control", "ctrl": "control",
    "healthy": "control", "0": "control",
    "schizophrenia": "schizophrenia", "schizophrenic": "schizophrenia",
    "scz": "schizophrenia", "1": "schizophrenia",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feature selection on the schizophrenia TRAIN CSV ONLY "
                    "(schz_train.csv; never pass schz_test.csv here).")
    p.add_argument("--input", required=True,
                   help="Path to the schizophrenia TRAINING CSV "
                        "(schz_train.csv only; NOT the test set).")
    p.add_argument("--output-dir", default="schizophrenia_artifacts",
                   help="Directory for all artifacts "
                        "(default: schizophrenia_artifacts).")
    p.add_argument("--target-genes", type=int, default=38,
                   help="Exact number of genes to select (default 38).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--particles", type=int, default=30,
                   help="Swarm size (deterministic pool width).")
    p.add_argument("--iterations", type=int, default=40,
                   help="Number of search refinement iterations.")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Stratified CV folds inside training data.")
    p.add_argument("--estimators", type=int, default=200,
                   help="RandomForest trees for fitness evaluation.")
    return p.parse_args()

def validate_input(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cols_lower = {c: str(c).strip().lower() for c in df.columns}
    group_col = None
    for c, lc in cols_lower.items():
        if lc == "group":
            group_col = c
            break
    if group_col is None:
        raise SystemExit("ERROR: input CSV must contain a 'Group' label column.")

    y_raw = df[group_col].astype(str).str.strip().str.lower()
    # Rows with a missing/blank label are invalid and must be dropped first.
    missing_mask = y_raw.isin(["", "nan", "none", "null", "na"]) | y_raw.isna()
    n_missing = int(missing_mask.sum())
    if n_missing:
        print(f"WARNING: dropping {n_missing} row(s) with missing/blank label.")
        df = df.loc[~missing_mask].reset_index(drop=True)
        y_raw = df[group_col].astype(str).str.strip().str.lower()
    if df.empty:
        raise SystemExit("ERROR: no rows with valid labels remain.")

    unknown = sorted(set(y_raw.unique()) - set(LABEL_ALIASES))
    if unknown:
        raise SystemExit(f"ERROR: invalid labels {unknown!r}; expected "
                         "control/schizophrenia (or 0/1).")

    y = y_raw.map(LABEL_ALIASES).map({"control": 0, "schizophrenia": 1})
    if y.isna().any():
        raise SystemExit("ERROR: NaN labels found after normalization.")

    feature_cols, drop_meta = [], []
    for c in df.columns:
        if c == group_col:
            drop_meta.append(c)
            continue
        if cols_lower[c] in META_COLUMNS:  # ID_REF handled case-insensitively
            drop_meta.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
        else:
            # try coercion; if everything fails, treat as metadata/junk
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() >= 0.5 * len(coerced) and coerced.notna().any():
                df[c] = coerced
                feature_cols.append(c)
            else:
                drop_meta.append(c)

    if not feature_cols:
        raise SystemExit("ERROR: no numeric feature columns found.")
    if y.nunique() < 2:
        raise SystemExit("ERROR: training data must contain both classes "
                         "(control and schizophrenia).")
    X = df[feature_cols].copy()
    # drop columns that are entirely missing
    all_na = [c for c in X.columns if X[c].isna().all()]
    if all_na:
        X = X.drop(columns=all_na)
    return X, y

def make_rf(args) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=args.estimators,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )

def cv_fitness(args, X: pd.DataFrame, y: pd.Series, rf_proto) -> float:
    """Stratified CV AUC of the given feature subset, training data only."""
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                          random_state=args.seed)
    scorer_rf = clone(rf_proto)
    scores = cross_val_score(scorer_rf, X.values, y.values, cv=skf, scoring="roc_auc")
    return float(np.mean(scores))

def deterministic_selection(args, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], List[dict]]:
    """Deterministic RF wrapper ranking; returns exactly target genes.

    RandomForest fitness/importances are estimated on stratified folds of the
    training data only. The final top-k subset receives a separate stratified
    CV ROC-AUC estimate. Ties are resolved by original CSV feature order.
    """
    rf_proto = make_rf(args)
    k = args.target_genes
    n, d = X.shape
    if k > d:
        raise SystemExit(f"ERROR: --target-genes ({k}) exceeds number of "
                         f"available features ({d}).")

    class_counts = y.value_counts()
    min_class = int(class_counts.min())
    if args.cv_folds < 2 or args.cv_folds > min_class:
        raise SystemExit(f"ERROR: --cv-folds must be between 2 and the "
                         f"smallest class count ({min_class}).")
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                          random_state=args.seed)
    importances = np.zeros(d, dtype=float)
    for train_idx, _ in skf.split(X.values, y.values):
        rf = clone(rf_proto)
        rf.fit(X.values[train_idx], y.values[train_idx])
        importances += rf.feature_importances_ / args.cv_folds

    # Stable tie-breaking: mergesort preserves input order for equal scores.
    ranked_idx = np.argsort(-importances, kind="stable")
    selected_idx = ranked_idx[:k]
    selected_genes = [str(X.columns[i]) for i in selected_idx]
    selected_X = X.iloc[:, selected_idx]
    scores = cross_val_score(clone(rf_proto), selected_X.values, y.values,
                             cv=skf, scoring="roc_auc", n_jobs=1)
    mean_auc = float(np.mean(scores))

    results = []
    for rank, idx in enumerate(ranked_idx, start=1):
        results.append({"rank": rank, "feature": str(X.columns[idx]),
                        "rf_mean_importance": float(importances[idx]),
                        "selected": bool(rank <= k),
                        "selected_subset_cv_auc": mean_auc if rank <= k else np.nan})
    print(f"[RF-CV] selected-subset stratified CV ROC-AUC="
          f"{mean_auc:.4f} +/- {float(np.std(scores)):.4f}")
    return selected_genes, results

def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit(f"ERROR: input file not found: {args.input}")
    if args.target_genes < 1:
        raise SystemExit("ERROR: --target-genes must be >= 1.")
    if args.estimators < 1 or args.particles < 1 or args.iterations < 1:
        raise SystemExit("ERROR: --estimators, --particles, and --iterations must be >= 1.")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading schizophrenia TRAIN CSV (train only, never the test set): "
          f"{args.input}")
    df = pd.read_csv(args.input)
    X, y = validate_input(df)
    print(f"Train samples={X.shape[0]}, features={X.shape[1]}, "
          f"classes={dict(y.value_counts())}")

    # Median imputer fitted ONLY on training data
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns,
                         index=X.index)

    scaler = StandardScaler().fit(X_imp.values)  # train-only fit
    X_scaled = pd.DataFrame(scaler.transform(X_imp.values),
                            columns=X.columns, index=X.index)

    selected_genes, log = deterministic_selection(args, X_scaled, y)
    print(f"Selected exactly {len(selected_genes)} genes (target="
          f"{args.target_genes}): {selected_genes}")

    # ---- artifacts ----
    with open(os.path.join(args.output_dir, "selected_genes.json"), "w") as f:
        json.dump({"selected_genes": selected_genes,
                   "n_selected": len(selected_genes),
                   "seed": args.seed, "note": "train-only selection; the "
                   "schizophrenia test set must load artifacts, never rerun "
                   "PSO"}, f, indent=2)

    dump(imputer, os.path.join(args.output_dir, "imputer.pkl"))
    dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))

    meta = {
        "n_train_samples": int(X.shape[0]),
        "n_features_total": int(X.shape[1]),
        "n_selected": len(selected_genes),
        "selected_genes": selected_genes,
        "label_mapping": {"control": 0, "schizophrenia": 1},
        "ignored_metadata_columns": sorted(META_COLUMNS),
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "target_genes": args.target_genes,
        "warning": "Run this script ONLY on the schizophrenia TRAIN CSV "
                   "(schz_train.csv). The final independent test must load "
                   "these artifacts and must NOT rerun PSO/feature selection.",
    }
    with open(os.path.join(args.output_dir, "feature_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pd.DataFrame({"rank": range(1, len(selected_genes) + 1),
                  "gene": selected_genes}).to_csv(
        os.path.join(args.output_dir, "selected_genes.csv"), index=False)

    pd.DataFrame(log).to_csv(os.path.join(args.output_dir, "pso_results.csv"),
                             index=False)

    # SNN training input: imputed + scaled train matrix restricted to selected genes
    X_sel = X_scaled[selected_genes].copy()
    X_sel.insert(0, "Group", y.values)  # 0=control, 1=schizophrenia
    X_sel.to_csv(os.path.join(args.output_dir, "snn_train_input.csv"), index=False)

    print("All artifacts saved. Feature order is stable/deterministic for a "
          "fixed --seed. Final independent schizophrenia test must load "
          "artifacts (selected_genes.json / imputer.pkl / "
          "feature_metadata.json) and NEVER rerun PSO.")

if __name__ == "__main__":
    main()
