#!/usr/bin/env python3
"""Deterministic MOPSO feature selection for expression CSV files.

The input must contain either a ``Group`` or ``genotype`` label column. Numeric
columns other than labels and metadata are treated as candidate genes. Missing
numeric values are median-imputed. The output directory receives:

* snn_input_selected_genes.csv: metadata, selected features, encoded label
* selected_genes.csv: ranked selected feature names and objective metrics
* pso_results.csv: Pareto archive and the chosen solution

Example: python PSO.py --input "Top - 100 - Bipolar-(2).csv" --seed 42
"""

import argparse
import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

METADATA_COLUMNS = ("sample_id", "ID_REF", "Group", "genotype", "treatment")


def _positive_int(value, name):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def dominates(a, b):
    """Return True if objective vector a Pareto-dominates b (minimization)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


class Particle:
    """A particle with a deterministic, position-derived fixed-size feature mask."""
    def __init__(self, n_features, max_selected_genes, rng):
        self.n_features = int(n_features)
        self.max_selected_genes = min(int(max_selected_genes), self.n_features)
        if self.max_selected_genes < 1:
            raise ValueError("At least one feature must be selectable.")
        self.position = rng.random(self.n_features)
        self.velocity = np.zeros(self.n_features, dtype=float)
        self.selected_count = int(rng.integers(1, self.max_selected_genes + 1))
        self.mask = self.make_mask()
        self.best_position = self.position.copy()
        self.best_mask = self.mask.copy()
        self.objectives = (float("inf"), float("inf"))
        self.best_objectives = (float("inf"), float("inf"))
        self.evaluated = False

    def make_mask(self):
        order = np.argsort(-self.position, kind="stable")[:self.selected_count]
        mask = np.zeros(self.n_features, dtype=bool)
        mask[order] = True
        return mask

    def refresh_mask(self):
        self.mask = self.make_mask()


class MOPSOFeatureSelection:
    def __init__(self, num_particles=25, max_iter=20, archive_size=30,
                 max_selected_genes=100, seed=42, n_estimators=30, cv_folds=3):
        self.num_particles = _positive_int(num_particles, "particles")
        self.max_iter = int(max_iter)
        if self.max_iter < 0:
            raise ValueError("iterations must be zero or greater.")
        self.archive_size = _positive_int(archive_size, "archive size")
        self.max_selected_genes = _positive_int(max_selected_genes, "max genes")
        self.seed = int(seed)
        self.n_estimators = _positive_int(n_estimators, "estimators")
        self.requested_cv_folds = _positive_int(cv_folds, "CV folds")
        self.rng = np.random.default_rng(self.seed)
        self.w, self.c1, self.c2 = 0.5, 1.5, 1.5
        self.cv_splits = None

    def evaluate(self, particle, X, y):
        """Evaluate exactly the particle's stored mask: [error, selected ratio]."""
        mask = particle.mask
        if mask.shape != (X.shape[1],):
            raise ValueError("Stored mask length does not match feature count.")
        n_selected = int(mask.sum())
        if not 1 <= n_selected <= min(self.max_selected_genes, X.shape[1]):
            raise ValueError("Selected feature count is outside the permitted range.")
        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.seed,
            n_jobs=-1, class_weight="balanced")
        scores = cross_val_score(classifier, X[:, mask], y, cv=self.cv_splits,
                                 scoring="accuracy", error_score="raise", n_jobs=1)
        particle.objectives = (1.0 - float(np.mean(scores)), n_selected / X.shape[1])
        particle.evaluated = True

    def update_archive(self, swarm, archive):
        combined = [copy.deepcopy(p) for p in archive]
        combined.extend(copy.deepcopy(p) for p in swarm if p.evaluated)
        front = []
        for i, particle in enumerate(combined):
            if any(i != j and dominates(other.objectives, particle.objectives)
                   for j, other in enumerate(combined)):
                continue
            if not any(np.allclose(particle.objectives, other.objectives) for other in front):
                front.append(particle)
        front.sort(key=lambda p: (p.objectives[0], p.objectives[1]))
        return front[:self.archive_size]

    def run(self, X, y):
        n_features = X.shape[1]
        class_counts = np.bincount(np.asarray(y, dtype=int))
        positive_counts = class_counts[class_counts > 0]
        if len(positive_counts) < 2 or int(positive_counts.min()) < 2:
            raise ValueError("At least two classes and two samples per class are required.")
        n_splits = min(self.requested_cv_folds, int(positive_counts.min()))
        if n_splits < 2:
            raise ValueError("Not enough samples per class for stratified cross-validation.")
        self.cv_splits = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                         random_state=self.seed)
        swarm = [Particle(n_features, self.max_selected_genes, self.rng)
                 for _ in range(self.num_particles)]
        for particle in swarm:
            self.evaluate(particle, X, y)
            particle.best_position = particle.position.copy()
            particle.best_mask = particle.mask.copy()
            particle.best_objectives = particle.objectives
        archive = self.update_archive(swarm, [])
        for iteration in range(1, self.max_iter + 1):
            leader = archive[int(self.rng.integers(len(archive)))]
            for particle in swarm:
                r1, r2 = self.rng.random(n_features), self.rng.random(n_features)
                particle.velocity = (self.w * particle.velocity
                                     + self.c1 * r1 * (particle.best_position - particle.position)
                                     + self.c2 * r2 * (leader.position - particle.position))
                particle.position = np.clip(particle.position + particle.velocity, 0.0, 1.0)
                particle.refresh_mask()
                self.evaluate(particle, X, y)
                if dominates(particle.objectives, particle.best_objectives):
                    particle.best_position = particle.position.copy()
                    particle.best_mask = particle.mask.copy()
                    particle.best_objectives = particle.objectives
            archive = self.update_archive(swarm, archive)
            counts = [int(p.mask.sum()) for p in archive]
            print(f"Iteration {iteration:02d}/{self.max_iter:02d} | Pareto solutions: "
                  f"{len(archive)} | features: {counts}")
        return archive


def _normalize_label_text(value):
    text = str(value).strip().casefold()
    return re.sub(r"[\s_\-]+", " ", text).strip()


def encode_labels(series, label_col):
    """Encode Group aliases or arbitrary genotype classes to stable integer labels."""
    raw = series.astype("string").str.strip()
    if raw.isna().any() or raw.eq("").any():
        raise ValueError(f"Label column {label_col!r} contains missing or blank values.")
    normalized = raw.map(_normalize_label_text)
    if label_col.casefold() == "group":
        # Canonical names also allow common abbreviations and punctuation variants.
        aliases = {
            "control": "control", "healthy control": "control", "healthy": "control",
            "bipolar": "bipolar", "bipolar disorder": "bipolar", "bd": "bipolar",
            "schizophrenia": "schizophrenia", "schizophrenic": "schizophrenia",
            "schz": "schizophrenia", "schizo": "schizophrenia", "sz": "schizophrenia",
            "schizoaffective": "schizoaffective",
            "schizoaffective disorder": "schizoaffective",
            "schizo affective": "schizoaffective", "schizo affective disorder": "schizoaffective",
            "schizoaffective disorder bipolar type": "schizoaffective",
            "schizoaffective disorder depressive type": "schizoaffective",
        }
        canonical = normalized.map(aliases)
        if canonical.isna().any():
            unknown = sorted(raw[canonical.isna()].astype(str).unique().tolist())
            raise ValueError(f"Unknown Group label(s): {unknown}. Supported groups include control, "
                             "bipolar, schizophrenia, and schizoaffective.")
        # Stable explicit ordering when multiple supported groups occur.
        order = [name for name in ("control", "bipolar", "schizophrenia", "schizoaffective")
                 if name in set(canonical)]
        encoded = canonical.map({name: i for i, name in enumerate(order)}).astype(np.int64)
    else:
        # Preserve semantic common genotype ordering, then append any other labels stably.
        values = sorted(normalized.unique().tolist())
        preferred_groups = [{"wt", "wild type", "wildtype"},
                            {"het akap11 ko", "het akap11 ko"}]
        order = []
        for group in preferred_groups:
            order.extend(value for value in values if value in group and value not in order)
        order.extend(value for value in values if value not in order)
        mapping = {name: i for i, name in enumerate(order)}
        encoded = normalized.map(mapping).astype(np.int64)
    if encoded.nunique() < 2:
        raise ValueError(f"Label column {label_col!r} must contain at least two classes.")
    counts = encoded.value_counts()
    if int(counts.min()) < 2:
        raise ValueError("Each class must have at least two samples for stratified CV.")
    return encoded.to_numpy(dtype=np.int64)


def _unique_clean_column_names(columns):
    """Strip headers and make empty/duplicate names unique without losing columns."""
    result, used = [], set()
    for i, raw_name in enumerate(columns):
        base = str(raw_name).strip() or f"unnamed_{i + 1}"
        candidate, suffix = base, 2
        while candidate in used:
            candidate = f"{base}.{suffix}"
            suffix += 1
        result.append(candidate)
        used.add(candidate)
    return result


def run_feature_selection(input_csv, output_dir, seed=42, particles=25, iterations=20,
                          max_genes=100, archive_size=30, n_estimators=30, cv_folds=3):
    """Run MOPSO, save all three CSV outputs, and return their paths and solution."""
    input_csv = Path(input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        raise ValueError("Input CSV has no data rows.")
    df.columns = _unique_clean_column_names(df.columns)
    lower_to_col = {}
    for col in df.columns:
        lower_to_col.setdefault(col.casefold(), col)
    label_col = lower_to_col.get("group") or lower_to_col.get("genotype")
    if label_col is None:
        raise ValueError("Input CSV must contain either a 'Group' or 'genotype' column.")
    y = encode_labels(df[label_col], label_col)

    # Only numeric-valued, non-metadata columns are candidate features.
    metadata_names = {name.casefold() for name in METADATA_COLUMNS}
    excluded = metadata_names | {label_col.casefold()}
    feature_frame = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col.casefold() in excluded:
            continue
        converted = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if converted.notna().any():
            feature_frame[col] = converted
    if feature_frame.shape[1] == 0:
        raise ValueError("No numeric feature columns found after excluding metadata and labels.")
    # Entirely missing values cannot be imputed; other missing values use column medians.
    feature_frame = feature_frame.dropna(axis=1, how="all")
    if feature_frame.shape[1] == 0:
        raise ValueError("No usable numeric feature columns remain (all values are missing).")
    feature_frame = feature_frame.fillna(feature_frame.median()).fillna(0.0)
    X = feature_frame.to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix still contains non-finite values after imputation.")
    if len(y) != X.shape[0]:
        raise ValueError("Label and feature row counts do not match.")
    if X.shape[0] < 4:
        raise ValueError("At least four samples are required for stratified cross-validation.")

    selector = MOPSOFeatureSelection(
        num_particles=particles, max_iter=iterations, archive_size=archive_size,
        max_selected_genes=max_genes, seed=seed, n_estimators=n_estimators,
        cv_folds=cv_folds)
    archive = selector.run(X, y)
    if not archive:
        raise RuntimeError("MOPSO completed without any Pareto solutions.")
    # Tie-break precisely as requested: minimum error, then minimum number of genes.
    chosen = min(archive, key=lambda p: (p.objectives[0], int(p.mask.sum()), p.objectives[1]))
    selected_indices = np.flatnonzero(chosen.mask)
    selected_features = [feature_frame.columns[i] for i in selected_indices]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep metadata in the requested order, case-insensitively, if present.
    metadata_cols = []
    for desired in METADATA_COLUMNS:
        actual = lower_to_col.get(desired.casefold())
        if actual is not None and actual not in metadata_cols:
            metadata_cols.append(actual)
    snn = df.loc[:, metadata_cols].copy()
    for feature in selected_features:
        snn[feature] = feature_frame[feature].to_numpy()
    snn["label_encoded"] = y
    snn_path = output_dir / "snn_input_selected_genes.csv"
    snn.to_csv(snn_path, index=False)

    chosen_error, chosen_ratio = chosen.objectives
    selected_rows = [{
        "rank": rank, "gene": gene, "objective_error": chosen_error,
        "selected_gene_ratio": chosen_ratio,
    } for rank, gene in enumerate(selected_features, start=1)]
    selected_path = output_dir / "selected_genes.csv"
    pd.DataFrame(selected_rows, columns=["rank", "gene", "objective_error",
                                         "selected_gene_ratio"]).to_csv(selected_path, index=False)

    result_rows = []
    for i, particle in enumerate(archive, start=1):
        genes = [feature_frame.columns[j] for j in np.flatnonzero(particle.mask)]
        result_rows.append({
            "solution_id": i,
            "objective_error": particle.objectives[0],
            "selected_gene_ratio": particle.objectives[1],
            "selected_gene_count": len(genes),
            "selected": bool(np.array_equal(particle.mask, chosen.mask)),
            "genes": ";".join(map(str, genes)),
        })
    results_path = output_dir / "pso_results.csv"
    pd.DataFrame(result_rows).to_csv(results_path, index=False)
    print(f"Selected {len(selected_features)} of {X.shape[1]} features; "
          f"CV error={chosen_error:.6f}; CV folds={min(cv_folds, int(pd.Series(y).value_counts().min()))}")
    print(f"Outputs written to: {output_dir.resolve()}")
    return {"snn_input": snn_path, "selected_genes": selected_path,
            "pso_results": results_path, "selected_features": selected_features,
            "objective_error": chosen_error, "selected_gene_ratio": chosen_ratio}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="Top - 100 - Schz .csv",
                        help="Input expression CSV (default: %(default)s)")
    parser.add_argument("--output-dir", default="pso_output",
                        help="Directory for the three output CSV files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--particles", type=int, default=25, help="Number of particles")
    parser.add_argument("--iterations", type=int, default=20, help="MOPSO iterations")
    parser.add_argument("--max-genes", type=int, default=100,
                        help="Maximum selected features (capped at available feature count)")
    parser.add_argument("--archive-size", type=int, default=30, help="Pareto archive limit")
    parser.add_argument("--estimators", type=int, default=30, help="Random forest trees")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Maximum stratified CV folds (capped by smallest class)")
    args = parser.parse_args(argv)
    try:
        run_feature_selection(args.input, args.output_dir, args.seed, args.particles,
                              args.iterations, args.max_genes, args.archive_size,
                              args.estimators, args.cv_folds)
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
