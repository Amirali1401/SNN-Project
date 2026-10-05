import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# تنظیمات
# ============================================================

TEST_FILE = "bipolar_test.csv"

ARTIFACT_DIR = "bipolar_artifacts"

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "bipolar_snn_model.keras"
)

SCALER_PATH = os.path.join(
    ARTIFACT_DIR,
    "bipolar_scaler.pkl"
)

GENES_PATH = os.path.join(
    ARTIFACT_DIR,
    "bipolar_selected_genes.json"
)

MODEL_INFO_PATH = os.path.join(
    ARTIFACT_DIR,
    "bipolar_model_info.json"
)

OUTPUT_FILE = "bipolar_test_predictions.csv"

DEFAULT_THRESHOLD = 0.5


# ============================================================
# بررسی فایل‌ها
# ============================================================

required_files = [
    TEST_FILE,
    MODEL_PATH,
    SCALER_PATH,
    GENES_PATH
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nفایل زیر پیدا نشد:\n{file_path}\n"
        )


# ============================================================
# بارگذاری مدل و preprocessing
# ============================================================

print("Loading model and preprocessing files...")

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

with open(
    GENES_PATH,
    "r",
    encoding="utf-8"
) as f:
    selected_genes = json.load(f)

threshold = DEFAULT_THRESHOLD
target_col_from_info = None

if os.path.exists(MODEL_INFO_PATH):
    with open(
        MODEL_INFO_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        model_info = json.load(f)

    threshold = model_info.get(
        "threshold",
        DEFAULT_THRESHOLD
    )

    target_col_from_info = model_info.get(
        "target_column",
        None
    )

print(f"تعداد ژن‌های استفاده‌شده: {len(selected_genes)}")
print(f"ژن‌ها: {selected_genes}")
print(f"Threshold: {threshold}")


# ============================================================
# بارگذاری داده تست
# ============================================================

df_test = pd.read_csv(TEST_FILE)

print(f"\nتعداد نمونه‌های تست: {len(df_test)}")
print(f"تعداد ستون‌های تست: {len(df_test.columns)}")

if "ID_REF" in df_test.columns:
    df_test = df_test.drop(
        columns=["ID_REF"]
    )


# ============================================================
# بررسی ژن‌های موردنیاز
# ============================================================

missing_genes = [
    gene for gene in selected_genes
    if gene not in df_test.columns
]

if missing_genes:
    raise ValueError(
        "\nژن‌های زیر در فایل تست وجود ندارند:\n"
        + "\n".join(missing_genes)
    )


# ============================================================
# آماده‌سازی X_test
# ============================================================

X_test_df = df_test[selected_genes].copy()

for gene in selected_genes:
    X_test_df[gene] = pd.to_numeric(
        X_test_df[gene],
        errors="coerce"
    )

if X_test_df.isnull().any().any():
    missing_values = X_test_df.isnull().sum().sum()

    raise ValueError(
        f"\nتعداد {missing_values} مقدار غیرعددی یا گمشده "
        "در ویژگی‌های تست وجود دارد."
    )

X_test = X_test_df.values

# استفاده از scaler ذخیره‌شده در زمان آموزش
X_test_scaled = scaler.transform(X_test)


# ============================================================
# پیش‌بینی
# ============================================================

print("\nRunning model prediction...")

y_prob = model.predict(
    X_test_scaled,
    verbose=0
).ravel()

y_pred = (
    y_prob >= threshold
).astype(int)


# ============================================================
# ذخیره نتایج پیش‌بینی
# ============================================================

results = df_test.copy()

results["predicted_probability"] = y_prob
results["predicted_label"] = y_pred
results["predicted_class"] = np.where(
    y_pred == 1,
    "Bipolar",
    "Control"
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nنتایج در فایل زیر ذخیره شدند:")
print(OUTPUT_FILE)

print("\nتعداد پیش‌بینی هر کلاس:")

print(
    results["predicted_class"].value_counts()
)

print("\nنمونه نتایج:")
print(
    results[
        [
            "predicted_probability",
            "predicted_label",
            "predicted_class"
        ]
    ].head(10)
)


# ============================================================
# پیدا کردن ستون Label واقعی
# ============================================================

possible_target_columns = []

if target_col_from_info is not None:
    possible_target_columns.append(
        target_col_from_info
    )

possible_target_columns.extend([
    "label",
    "Label",
    "target",
    "Target",
    "class",
    "Class",
    "diagnosis",
    "Diagnosis",
    "group",
    "Group"
])

actual_target_col = None

for column in possible_target_columns:
    if column in df_test.columns:
        actual_target_col = column
        break


# ============================================================
# ارزیابی در صورت وجود Label واقعی
# ============================================================

if actual_target_col is None:

    print(
        "\nستون برچسب واقعی در فایل تست پیدا نشد."
    )

    print(
        "بنابراین فقط پیش‌بینی‌ها تولید شدند."
    )

else:

    print(
        f"\nستون برچسب واقعی: {actual_target_col}"
    )

    y_test_raw = df_test[
        actual_target_col
    ]

    # اگر labelها عددی باشند
    if pd.api.types.is_numeric_dtype(
        y_test_raw
    ):
        y_test = y_test_raw.astype(int).values

    # اگر labelها متنی باشند
    else:

        normalized_labels = (
            y_test_raw
            .astype(str)
            .str.strip()
            .str.lower()
        )

        label_mapping = {
            "control": 0,
            "healthy": 0,
            "normal": 0,
            "hc": 0,

            "bipolar": 1,
            "bd": 1,
            "patient": 1,
            "case": 1
        }

        unknown_labels = set(
            normalized_labels
        ) - set(label_mapping.keys())

        if unknown_labels:
            raise ValueError(
                "\nبرچسب‌های ناشناخته در فایل تست:\n"
                + str(unknown_labels)
            )

        y_test = (
            normalized_labels
            .map(label_mapping)
            .astype(int)
            .values
        )

    if len(np.unique(y_test)) < 2:

        print(
            "\nفایل تست فقط شامل یک کلاس است."
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")

    else:

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = matrix.ravel()

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        print("\n" + "=" * 60)
        print("FINAL BIPOLAR TEST RESULTS")
        print("=" * 60)
        print(f"Accuracy    : {accuracy:.4f}")
        print(f"Precision   : {precision:.4f}")
        print(f"Recall      : {recall:.4f}")
        print(f"F1-Score    : {f1:.4f}")
        print(f"ROC-AUC     : {auc:.4f}")
        print(f"Sensitivity : {sensitivity:.4f}")
        print(f"Specificity : {specificity:.4f}")
        print("=" * 60)

        print("\nConfusion Matrix:")
        print(matrix)

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "Control",
                    "Bipolar"
                ],
                zero_division=0
            )
        )

        results["actual_label"] = y_test

        results["correct_prediction"] = (
            y_test == y_pred
        )

        results.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

print("\nTest process completed successfully.")
