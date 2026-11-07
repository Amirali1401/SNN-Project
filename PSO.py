#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature selection (deterministic RF wrapper) on TRAIN ONLY, 3-class:
    control / mdd / bipolar
Input: preprocessed_data.npz  (X_train: scaled, y_train: 0/1/2)
Feature names are read from merged_MDD_Bipolar_100probes_labeled.csv
(must be in the same folder; column order must match NPZ).
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROTECTED = {"Sample_ID", "Group", "Source"}
LABEL_MAP = {"bipolar": 0, "control": 1, "mdd": 2}


def parse_args():
    p = argparse.ArgumentParser(description="3-class feature selection, train-only")
    p.add_argument("--npz", default="preprocessed_data.npz")
    p.add_argument("--csv", default="merged_MDD_Bipolar_100probes_labeled.csv",
                   help="CSV only used to recover feature names (NOT for fitting).")
    p.add_argument("--output-dir", default="fs_artifacts")
    p.add_argument("--target-features", type=int, default=38)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--estimators", type=int, default=300)
    p.add_argument("--scoring", default="roc_auc_ovr")
    return p.parse_args()


def make_rf(seed: int, estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=estimators, random_state=seed, n_jobs=-1,
        class_weight="balanced")


def deterministic_selection(args, X: np.ndarray, y: np.ndarray,
                            feature_names: List[str]):
    rf_proto = make_rf(args.seed, args.estimators)
    k = args.target_features
    n, d = X.shape
    if k > d:
        raise SystemExit(f"ERROR: target ({k}) > features ({d}).")

    class_counts = pd.Series(y).value_counts()
    min_class = int(class_counts.min())
    if args.cv_folds < 2 or args.cv_folds > min_class:
        raise SystemExit(f"ERROR: cv-folds must be 2..min-class({min_class}).")

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                          random_state=args.seed)
    importances = np.zeros(d, dtype=float)
    for train_idx, _ in skf.split(X, y):
        rf = clone(rf_proto)
        rf.fit(X[train_idx], y[train_idx])
        importances += rf.feature_importances_ / args.cv_folds

    ranked_idx = np.argsort(-importances, kind="stable")
    selected_idx = ranked_idx[:k]
    selected_features = [feature_names[i] for i in selected_idx]

    mean_score, std_score, scores = None, None, None
    if X.shape[0] >= 2 * k:  # subset CV only if enough samples
        X_sel = X[:, selected_idx]
        scores = cross_val_score(clone(rf_proto), X_sel, y, cv=skf,
                                 scoring=args.scoring, n_jobs=1)
        mean_score, std_score = float(np.mean(scores)), float(np.std(scores))
        print(f"[RF-CV] selected-subset CV {args.scoring} = "
              f"{mean_score:.4f} +/- {std_score:.4f}")

    log = []
    for rank, idx in enumerate(ranked_idx, start=1):
        log.append({
            "rank": rank,
            "feature": feature_names[idx],
            "rf_mean_importance": float(importances[idx]),
            "selected": bool(rank <= k),
            "selected_subset_cv_score": mean_score if (rank <= k and mean_score is not None) else np.nan
        })
    return selected_features, log


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]

    df = pd.read_csv(args.csv)
    feature_names = [c for c in df.columns if c not in PROTECTED]
    if X_train.shape[1] != len(feature_names):
        raise SystemExit("ERROR: NPZ feature count != CSV feature count "
                         "(check column order).")

    # Validate y is 3-class
    classes = sorted(pd.Series(y_train).unique())
    print("y classes found:", classes)

    print(f"Train samples={X_train.shape[0]}, features={X_train.shape[1]}")
    print("Class counts:", dict(pd.Series(y_train).value_counts()))

    selected, log = deterministic_selection(args, X_train, y_train, feature_names)
    print(f"Selected exactly {len(selected)} features.")

    with open(os.path.join(args.output_dir, "selected_features.json"), "w") as f:
        json.dump({
            "label_mapping": LABEL_MAP,
            "selected_features": selected,
            "n_selected": len(selected),
            "seed": args.seed,
            "cv_folds": args.cv_folds,
            "scoring": args.scoring,
            "classes": sorted(set(map(int, y_train))),
        }, f, indent=2)

    pd.DataFrame({"rank": range(1, len(selected) + 1),
                  "feature": selected}).to_csv(
        os.path.join(args.output_dir, "selected_features.csv"), index=False)
    pd.DataFrame(log).to_csv(os.path.join(args.output_dir, "fs_results.csv"),
                             index=False)

    sel_idx = [feature_names.index(f) for f in selected]
    np.savez(os.path.join(args.output_dir, "train_selected.npz"),
             X_train_selected=X_train[:, sel_idx],
             y_train=y_train,
             selected_features=np.array(selected, dtype=object))
    print("All artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
