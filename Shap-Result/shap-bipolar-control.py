
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

warnings.filterwarnings("ignore")

# =========================================================
# ۱. تنظیمات مسیرها (همه به صورت Path آبجکت با r-string)
# =========================================================
BASE_DIR = Path(__file__).parent

# حتماً دور تمام این مسیرها تابع Path(...) قرار داشته باشد
TEST_PATH = Path(r"C:\Users\LENO\Desktop\deep learning\Top - 100 - Bipolar.csv")
BACKGROUND_PATH = Path(r"C:\Users\LENO\Desktop\deep learning\snn_train_input.csv")
OUTPUT_DIR = Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_shap_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BACKGROUND_SIZE = 30
TARGET_CLASS_INDEX = 1


# =========================================================
# ۲. بارگذاری دقیق و هوشمند مصنوعات (با مسیرهای r-string)
# =========================================================
print(f"در حال جستجو و بارگذاری مصنوعات در: {BASE_DIR.resolve()}")

# الف) فایل ژن‌های برگزیده (۳۸ ژن دوقطبی)
gene_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\bipolar_selected_genes.json"),
    BASE_DIR / "selected_genes-(2).json",
    BASE_DIR / "selected_genes.csv"
]
selected_genes_path = next((p for p in gene_candidates if p.exists()), None)

if selected_genes_path is None:
    raise FileNotFoundError("فایل ژن‌های برگزیده دوقطبی یافت نشد.")

print(f"-> فایل ژن‌ها از این مسیر بارگذاری شد: {selected_genes_path}")

if selected_genes_path.suffix == ".json":
    with open(selected_genes_path, "r", encoding="utf-8") as f:
        loaded_genes = json.load(f)
    if isinstance(loaded_genes, dict):
        selected_genes = (
            loaded_genes.get("selected_genes")
            or loaded_genes.get("features")
            or loaded_genes.get("genes")
        )
    else:
        selected_genes = list(loaded_genes)
else:
    df_genes = pd.read_csv(selected_genes_path)
    col = next((c for c in ["gene", "probe", "features"] if c in df_genes.columns), df_genes.columns[-1])
    selected_genes = df_genes[col].astype(str).tolist()

print(f"تعداد ژن‌های بارگذاری‌شده: {len(selected_genes)}")

# ب) مسیرهای مدل، اسکیلر و ایمپیتر با r-string
model_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\bipolar_snn_model.keras"),
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\snn_model.keras"),
    BASE_DIR / "snn_model.keras",
]
scaler_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\scaler-(2).pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\scaler-(2).pkl"),
    BASE_DIR / "scaler-(2).pkl",
    BASE_DIR / "scaler.pkl",
]
imputer_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\imputer-(2).pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\imputer-(2).pkl"),
    BASE_DIR / "imputer-(2).pkl",
    BASE_DIR / "imputer.pkl",
]

model_path = next((p for p in model_candidates if p.exists()), None)
scaler_path = next((p for p in scaler_candidates if p.exists()), None)
imputer_path = next((p for p in imputer_candidates if p.exists()), None)

if not model_path:
    raise FileNotFoundError("فایل مدل snn_model.keras پیدا نشد.")

print(f"-> فایل مدل بارگذاری شد: {model_path}")
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None
imputer = joblib.load(imputer_path) if imputer_path and imputer_path.exists() else None

# =========================================================
# ۳. بارگذاری و پیش‌پردازش داده‌های تست
# =========================================================
if not TEST_PATH.exists():
    raise FileNotFoundError(f"فایل داده تست پیدا نشد: {TEST_PATH}")

df_test = pd.read_csv(TEST_PATH)
print(f"ابعاد داده تست بارگذاری‌شده: {df_test.shape}")

# بررسی وجود ژن‌ها در داده تست
missing_genes = [g for g in selected_genes if g not in df_test.columns]
if missing_genes:
    raise KeyError(f"این ژن‌ها در داده تست موجود نیستند: {missing_genes}")

X_raw = df_test[selected_genes].copy().apply(pd.to_numeric, errors="coerce")

if imputer is not None:
    X_imputed = imputer.transform(X_raw)
else:
    X_imputed = X_raw.fillna(X_raw.median()).values

if scaler is not None:
    X_processed = scaler.transform(X_imputed)
else:
    X_processed = X_imputed

X_processed = np.asarray(X_processed, dtype=np.float32)

# =========================================================
# ۴. تعریف تابع پیش‌بینی و محاسبه مقادیر SHAP
# =========================================================
def model_predict(data):
    preds = model.predict(data, verbose=0)
    if preds.shape[1] == 1:
        return np.hstack([1 - preds, preds])
    return preds

if BACKGROUND_PATH and BACKGROUND_PATH.exists():
    df_bg = pd.read_csv(BACKGROUND_PATH)
    X_bg_raw = df_bg[[g for g in selected_genes if g in df_bg.columns]].copy().apply(pd.to_numeric, errors="coerce")
    X_bg_imp = imputer.transform(X_bg_raw) if imputer else X_bg_raw.fillna(X_bg_raw.median()).values
    X_bg = scaler.transform(X_bg_imp) if scaler else X_bg_imp
    background_data = shap.maskers.Independent(np.asarray(X_bg, dtype=np.float32), max_samples=BACKGROUND_SIZE)
else:
    background_data = shap.maskers.Independent(X_processed[:BACKGROUND_SIZE], max_samples=BACKGROUND_SIZE)

print("در حال محاسبه مقادیر SHAP برای مدل دوقطبی (لطفاً کمی صبر کنید)...")
explainer = shap.KernelExplainer(model_predict, background_data)
shap_values = explainer.shap_values(X_processed, nsamples="auto")

if isinstance(shap_values, list):
    sv_target = shap_values[TARGET_CLASS_INDEX]
else:
    sv_target = shap_values

# =========================================================
# ۵. ذخیره و چاپ نتایج رتبه‌بندی ژن‌ها
# =========================================================
mean_abs_shap = np.mean(np.abs(sv_target), axis=0)
gene_importance_df = pd.DataFrame({
    "gene": selected_genes,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

gene_importance_df["rank"] = gene_importance_df.index + 1

importance_csv_path = OUTPUT_DIR / "bipolar_shap_feature_importance.csv"
gene_importance_df.to_csv(importance_csv_path, index=False, encoding="utf-8-sig")

print("\n--- ۱۰ ژن برتر تأثیرگذار در مدل دوقطبی (بر اساس SHAP) ---")
print(gene_importance_df.head(10))
print(f"\n📁 نتایج با موفقیت ذخیره شد در مسیر:\n{OUTPUT_DIR.resolve()}")
