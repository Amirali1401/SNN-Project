import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import AlphaDropout, Dense
from tensorflow.keras.models import Sequential


import json
import os
import re

# مسیر پوشه‌ای که train-bipolar.py داخل آن قرار دارد
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# نام واقعی فایل CSV داده‌های بیان ژن را اینجا قرار بده
DATA_PATH = os.path.join(BASE_DIR, 'bipolar_train.csv')

# فایل selected_genes.csv در کنار فایل train-bipolar.py
SELECTED_GENES_PATH = os.path.join(BASE_DIR, 'bipolar_pso_output/selected_genes.csv')

ARTIFACT_DIR = os.path.join(BASE_DIR, 'bipolar_artifacts')

RANDOM_STATE = 42
EPOCHS = 80
BATCH_SIZE = 8
N_SPLITS = 5

np.random.seed(RANDOM_STATE)
try:
    tf.random.set_seed(RANDOM_STATE)
except Exception:
    pass


def _clean_gene_id(value):
    gene = str(value).strip()
    gene = re.sub(r'^\s*\d+\s*[:,]\s*', '', gene).strip()
    return gene


def load_selected_genes(path):
    if not os.path.isfile(path):
        raise FileNotFoundError('Selected-gene file not found: {}'.format(path))

    genes_df = pd.read_csv(path, dtype=str)
    genes_df.columns = [str(column).strip() for column in genes_df.columns]
    accepted = ('probe_id', 'gene', 'gene_symbol', 'selected_gene')
    selected_column = next((name for name in accepted if name in genes_df.columns), None)
    if selected_column is None:
        if genes_df.shape[1] == 1:
            selected_column = genes_df.columns[0]
        else:
            raise ValueError(
                'Selected-gene CSV must contain one of {} or have exactly one column; found: {}'
                .format(', '.join(accepted), genes_df.columns.tolist())
            )

    genes = []
    seen = set()
    for value in genes_df[selected_column].dropna():
        gene = _clean_gene_id(value)
        if gene and gene not in seen:
            genes.append(gene)
            seen.add(gene)
    if not genes:
        raise ValueError('No selected gene identifiers were found in {}'.format(path))
    return genes


def build_snn_model(input_dim):
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(64, activation='selu', kernel_initializer='lecun_normal'),
        AlphaDropout(0.1),
        Dense(32, activation='selu', kernel_initializer='lecun_normal'),
        AlphaDropout(0.1),
        Dense(16, activation='selu', kernel_initializer='lecun_normal'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mean_metric(fold_metrics, name):
    values = [item[name] for item in fold_metrics if item[name] is not None]
    return float(np.mean(values)) if values else None


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError('Training data file not found: {}'.format(DATA_PATH))
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(column).strip() for column in df.columns]
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError('Duplicate dataframe column names after stripping whitespace: {}'.format(duplicated))
    if 'Group' not in df.columns:
        raise ValueError("Training data must contain a 'Group' target column.")
    if 'ID_REF' in df.columns:
        df = df.drop(columns=['ID_REF'])

    selected_genes = load_selected_genes(SELECTED_GENES_PATH)
    missing = [gene for gene in selected_genes if gene not in df.columns]
    if missing:
        raise ValueError(
            'Selected gene IDs do not match training-data columns ({} missing). Examples: {}. '
            'Check that the selected file contains probe IDs, not gene symbols.'
            .format(len(missing), missing[:10])
        )

    try:
        feature_frame = df.loc[:, selected_genes].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Selected feature columns must contain numeric values: {}'.format(exc)) from exc
    if feature_frame.isna().any().any():
        bad_columns = feature_frame.columns[feature_frame.isna().any()].tolist()
        raise ValueError('Selected feature columns contain missing values: {}'.format(bad_columns[:10]))

    target = df['Group']
    if target.isna().any():
        raise ValueError("Target column 'Group' contains missing values.")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)
    if len(label_encoder.classes_) != 2:
        raise ValueError(
            "Target column 'Group' must contain exactly two classes; found {}: {}"
            .format(len(label_encoder.classes_), [_json_value(v) for v in label_encoder.classes_])
        )
    class_counts = np.bincount(y)
    if class_counts.min() < N_SPLITS:
        raise ValueError(
            'Each target class must have at least {} samples for StratifiedKFold; counts are {}.'
            .format(N_SPLITS, class_counts.tolist())
        )

    X = feature_frame.to_numpy(dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    print('Training SNN with {} selected features'.format(len(selected_genes)))
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_snn_model(input_dim=X_train.shape[1])
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        probabilities = model.predict(X_val, verbose=0).ravel()
        predictions = (probabilities >= 0.5).astype(int)
        auc = float(roc_auc_score(y_val, probabilities)) if np.unique(y_val).size == 2 else None
        metrics = {
            'accuracy': float(accuracy_score(y_val, predictions)),
            'roc_auc': auc,
            'precision': float(precision_score(y_val, predictions, zero_division=0)),
            'recall': float(recall_score(y_val, predictions, zero_division=0)),
            'f1': float(f1_score(y_val, predictions, zero_division=0)),
        }
        fold_metrics.append(metrics)
        print('Fold {}: {}'.format(fold, ', '.join(
            '{}={}'.format(key, 'NA' if value is None else '{:.4f}'.format(value))
            for key, value in metrics.items()
        )))

    mean_metrics = {name: _mean_metric(fold_metrics, name)
                    for name in ('accuracy', 'roc_auc', 'precision', 'recall', 'f1')}
    print('Mean CV metrics: {}'.format(', '.join(
        '{}={}'.format(key, 'NA' if value is None else '{:.4f}'.format(value))
        for key, value in mean_metrics.items()
    )))

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_model = build_snn_model(input_dim=X_scaled.shape[1])
    final_model.fit(X_scaled, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    model_path = os.path.join(ARTIFACT_DIR, 'bipolar_snn_model.keras')
    scaler_path = os.path.join(ARTIFACT_DIR, 'bipolar_scaler.pkl')
    genes_path = os.path.join(ARTIFACT_DIR, 'bipolar_selected_genes.json')
    metadata_path = os.path.join(ARTIFACT_DIR, 'bipolar_metadata.json')
    final_model.save(model_path)
    joblib.dump(final_scaler, scaler_path)
    with open(genes_path, 'w', encoding='utf-8') as output:
        json.dump(selected_genes, output, indent=2, ensure_ascii=True)
    metadata = {
        'data_path': DATA_PATH,
        'selected_gene_count': len(selected_genes),
        'target_column': 'Group',
        'target_mapping': {
            str(_json_value(label)): int(index)
            for index, label in enumerate(label_encoder.classes_)
        },
        'feature_order': selected_genes,
        'cv_metrics': {'folds': fold_metrics, 'mean': mean_metrics},
    }
    with open(metadata_path, 'w', encoding='utf-8') as output:
        json.dump(metadata, output, indent=2, ensure_ascii=True, allow_nan=False)
    print('Saved model, scaler, selected genes, and metadata to {}'.format(ARTIFACT_DIR))


if __name__ == '__main__':
    main()
