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

TEST_FILE = "schizophrenia_test.csv"

ARTIFACT_DIR = "schizophrenia_artifacts"

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "schizophrenia_snn_model.keras"
)

SCALER_PATH = os.path.join(
    ARTIFACT_DIR,
    "schizophrenia_scaler.pkl"
)

GENES_PATH = os.path.join(
    ARTIFACT_DIR,
    "schizophrenia_selected_genes.json"
)

MODEL_INFO_PATH = os.path.join(
    ARTIFACT_DIR,
    "schizophrenia_model_info.json"
)

OUTPUT_FILE = "schizophrenia_test_predictions.csv"

THRESHOLD = 0.5

# ============================================================
# بارگذاری مدل، scaler و اطلاعات ویژگی‌ها
# ============================================================

print("Loading model and preprocessing artifacts...")

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"فایل تست پیدا نشد: {TEST_FILE}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"مدل پیدا نشد: {MODEL_PATH}"
    )

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"Scaler پیدا نشد: {SCALER_PATH}"
    )

if not os.path.exists(GENES_PATH):
    raise FileNotFoundError(
        f"فایل ژن‌ها پیدا نشد: {GENES_PATH}"
    )

# بارگذاری مدل
model = load_model(MODEL_PATH)

# بارگذاری scaler
scaler = joblib.load(SCALER_PATH)

# بارگذاری ژن‌های انتخاب‌شده
with open(GENES_PATH, "r", encoding="utf-8") as f:
    selected_genes = json.load(f)

print(f"تعداد ژن‌های مورد استفاده: {len(selected_genes)}")
print(f"ژن‌ها: {selected_genes}")

# بارگذاری اطلاعات مدل، در صورت وجود
model_info = {}

if os.path.exists(MODEL_INFO_PATH):
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        model_info = json.load(f)

target_col_from_info = model_info.get("target_column", None)
threshold = model_info.get("threshold", THRESHOLD)

# ============================================================
# بارگذاری داده تست
# ============================================================

df_test = pd.read_csv(TEST_FILE)

print(f"\nتعداد نمونه‌های تست: {len(df_test)}")
print(f"تعداد ستون‌های تست: {len(df_test.columns)}")

# حذف ID_REF در صورت وجود
if "ID_REF" in df_test.columns:
    df_test = df_test.drop(columns=["ID_REF"])

# بررسی وجود ژن‌ها
missing_genes = [
    gene for gene in selected_genes
    if gene not in df_test.columns
]

if missing_genes:
    raise ValueError(
        "برخی از ژن‌های انتخاب‌شده در فایل تست وجود ندارند:\n"
        + "\n".join(missing_genes)
    )

# استخراج ویژگی‌ها با همان ترتیب زمان آموزش
X_test = df_test[selected_genes].copy()

# تبدیل مقادیر به عدد
for gene in selected_genes:
    X_test[gene] = pd.to_numeric(
        X_test[gene],
        errors="coerce"
    )

# بررسی مقادیر گمشده
if X_test.isnull().any().any():
    missing_count = X_test.isnull().sum().sum()

    raise ValueError(
        f"در داده تست {missing_count} مقدار غیرعددی یا گمشده وجود دارد."
    )

X_test = X_test.values

# استفاده از scaler آموزش‌دیده
X_test_scaled = scaler.transform(X_test)

# ============================================================
# پیش‌بینی
# ============================================================

print("\nRunning predictions...")

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
    "Schizophrenia",
    "Control"
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nنتایج پیش‌بینی ذخیره شد در:")
print(OUTPUT_FILE)

print("\nتعداد پیش‌بینی‌ها:")
print(pd.Series(y_pred).value_counts().sort_index())

print("\nنمونه نتایج:")
print(
    results[
        ["predicted_probability", "predicted_label", "predicted_class"]
    ].head(10)
)

# ============================================================
# ارزیابی، فقط در صورتی که Label واقعی موجود باشد
# ============================================================

possible_target_columns = []

if target_col_from_info is not None:
    possible_target_columns.append(target_col_from_info)

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

for col in possible_target_columns:
    if col in df_test.columns:
        actual_target_col = col
        break

if actual_target_col is None:
    print(
        "\nستون Label واقعی در فایل تست وجود ندارد."
    )
    print(
        "بنابراین فقط probability و predicted_label تولید شد."
    )

else:
    print(f"\nستون Label واقعی: {actual_target_col}")

    y_test_raw = df_test[actual_target_col]

    # فرض این کد این است که labelهای تست قبلاً 0 و 1 هستند.
    # در صورت وجود label متنی، آن‌ها را تبدیل می‌کنیم.
    if pd.api.types.is_numeric_dtype(y_test_raw):
        y_test = y_test_raw.astype(int).values

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
            "schizophrenia": 1,
            "schz": 1,
            "patient": 1,
            "case": 1
        }

        unknown_labels = set(normalized_labels) - set(label_mapping)

        if unknown_labels:
            raise ValueError(
                "برچسب‌های ناشناخته در فایل تست:\n"
                + str(unknown_labels)
            )

        y_test = normalized_labels.map(label_mapping).astype(int).values

    # بررسی وجود هر دو کلاس
    unique_classes = np.unique(y_test)

    if len(unique_classes) < 2:
        print(
            "\nهشدار: فایل تست فقط شامل یک کلاس است."
        )
        print(
            "ROC-AUC و بعضی معیارها قابل محاسبه نیستند."
        )

        accuracy = accuracy_score(y_test, y_pred)
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

        print(f"\nAccuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")

    else:
        # محاسبه ماتریس درهم‌ریختگی
        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        ).ravel()

        accuracy = accuracy_score(y_test, y_pred)

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

        auc = roc_auc_score(y_test, y_prob)

        ppv = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        npv = (
            tn / (tn + fn)
            if (tn + fn) > 0
            else 0.0
        )

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        print("\n" + "=" * 60)
        print("FINAL TEST RESULTS")
        print("=" * 60)
        print(f"Accuracy    : {accuracy:.4f}")
        print(f"Precision   : {precision:.4f}")
        print(f"Recall      : {recall:.4f}")
        print(f"F1-Score    : {f1:.4f}")
        print(f"ROC-AUC     : {auc:.4f}")
        print(f"PPV         : {ppv:.4f}")
        print(f"NPV         : {npv:.4f}")
        print(f"Sensitivity : {sensitivity:.4f}")
        print(f"Specificity : {specificity:.4f}")
        print("=" * 60)

        print("\nConfusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )
        )

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "Control",
                    "Schizophrenia"
                ],
                zero_division=0
            )
        )

        # اضافه کردن label واقعی به فایل خروجی
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
