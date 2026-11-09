import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

# ۱. تنظیمات اولیه و مسیر فایل‌ها
ARTIFACT_DIR = Path("scz_model_artifacts")
if not ARTIFACT_DIR.exists():
    ARTIFACT_DIR = Path(".")

DATA_PATH = Path("Top - 100 - Schz .csv")
BACKGROUND_PATH = None  # در صورت وجود فایل دادهٔ آموزش جداگانه، مسیر آن را اینجا قرار دهید
OUTPUT_DIR = Path("scz_shap_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND_SIZE = 50
EXPLAIN_SIZE = None     # برای تحلیل همهٔ نمونه‌ها None بماند
NSAMPLES = "auto"
RANDOM_SEED = 42
TARGET_CLASS_INDEX = 1  # کلاس ۱ معادل اسکیزوفرنی (بر اساس label_mapping.json)

# ۲. بارگذاری مصنوعات مدل اسکیزوفرنی
with open(ARTIFACT_DIR / "selected_genes.json", "r", encoding="utf-8") as f:
    selected_genes = json.load(f)
    if isinstance(selected_genes, dict) and "selected_genes" in selected_genes:
        selected_genes = selected_genes["selected_genes"]

imputer = joblib.load(ARTIFACT_DIR / "imputer.pkl")
scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
model = tf.keras.models.load_model(ARTIFACT_DIR / "snn_model.keras")

label_mapping = {}
label_map_path = ARTIFACT_DIR / "label_mapping.json"
if label_map_path.exists():
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

print(f"تعداد ژن‌های انتخاب‌شده برای اسکیزوفرنی: {len(selected_genes)}")

# ۳. تابع پیش‌پردازش دقیقاً مطابق پایپ‌لاین آموزش
def preprocess(df: pd.DataFrame) -> np.ndarray:
    missing_genes = [g for g in selected_genes if g not in df.columns]
    if missing_genes:
        raise KeyError(f"این ژن‌ها در فایل داده یافت نشدند: {missing_genes}")
    
    X = df[selected_genes].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    return np.asarray(X_scaled, dtype=np.float32)

# ۴. بارگذاری داده‌ها و ساخت دادهٔ مرجع (Background)
df_data = pd.read_csv(DATA_PATH)
X_all = preprocess(df_data)

rng = np.random.default_rng(RANDOM_SEED)

if BACKGROUND_PATH is not None and Path(BACKGROUND_PATH).exists():
    df_bg = pd.read_csv(BACKGROUND_PATH)
    X_bg_pool = preprocess(df_bg)
else:
    print("هشدار: فایل مرجع جداگانه معرفی نشده است؛ دادهٔ مرجع به‌صورت تصادفی از فایل فعلی انتخاب می‌شود.")
    X_bg_pool = X_all

bg_size = min(BACKGROUND_SIZE, len(X_bg_pool))
bg_indices = rng.choice(len(X_bg_pool), size=bg_size, replace=False)
X_background = X_bg_pool[bg_indices]

if EXPLAIN_SIZE is not None and EXPLAIN_SIZE < len(X_all):
    explain_indices = rng.choice(len(X_all), size=EXPLAIN_SIZE, replace=False)
    X_explain = X_all[explain_indices]
else:
    X_explain = X_all

# ۵. تعریف تابع پیش‌بینی تک‌خروجی برای جلوگیری از ابهام ابعاد
def predict_target(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    preds = np.asarray(model.predict(x_arr, verbose=0))
    if preds.ndim == 1:
        return preds
    if preds.ndim == 2:
        if preds.shape[1] == 1:
            return preds[:, 0]
        return preds[:, TARGET_CLASS_INDEX]
    raise ValueError(f"ابعاد خروجی مدل پشتیبانی نمی‌شود: {preds.shape}")

# ۶. اجرای تحلیل SHAP
explainer = shap.KernelExplainer(predict_target, X_background)
shap_values = explainer.shap_values(X_explain, nsamples=NSAMPLES, l1_reg=0.0)
shap_values = np.asarray(shap_values, dtype=np.float64)

if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
    shap_values = shap_values[:, :, 0]
elif shap_values.ndim != 2:
    raise ValueError(f"ابعاد مقادیر SHAP غیرمنتظره است: {shap_values.shape}")

# ۷. رتبه‌بندی ژن‌ها، کنترل جمع‌پذیری و ذخیرهٔ نتایج
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
mean_signed_shap = np.mean(shap_values, axis=0)

ranking_df = pd.DataFrame({
    "gene": selected_genes,
    "mean_abs_shap": mean_abs_shap,
    "mean_signed_shap": mean_signed_shap,
}).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)
ranking_df = ranking_df[["rank", "gene", "mean_abs_shap", "mean_signed_shap"]]

base_value = np.asarray(explainer.expected_value).reshape(-1)[0]
reconstructed = base_value + np.sum(shap_values, axis=1)
actual_preds = predict_target(X_explain)
max_additivity_error = float(np.max(np.abs(reconstructed - actual_preds)))

print(f"\nبیشترین خطای بازسازی خروجی مدل از SHAP: {max_additivity_error:.6e}\n")
print("رتبه‌بندی ۱۰ ژن برتر در مدل اسکیزوفرنی:")
print(ranking_df.to_string(index=False))

# ذخیرهٔ جدول رتبه‌بندی و مقادیر کامل به تفکیک نمونه
ranking_df.to_csv(OUTPUT_DIR / "scz_shap_feature_importance.csv", index=False)

shap_samples_df = pd.DataFrame(shap_values, columns=selected_genes)
if "ID_REF" in df_data.columns and len(shap_samples_df) == len(df_data):
    shap_samples_df.insert(0, "ID_REF", df_data["ID_REF"].values)
if "Group" in df_data.columns and len(shap_samples_df) == len(df_data):
    shap_samples_df.insert(1 if "ID_REF" in shap_samples_df.columns else 0, "Group", df_data["Group"].values)

shap_samples_df.to_csv(OUTPUT_DIR / "scz_shap_values_per_sample.csv", index=False)

metadata_summary = {
    "task": "schizophrenia_binary_classification",
    "num_genes": len(selected_genes),
    "num_explained_samples": int(len(X_explain)),
    "background_size": int(len(X_background)),
    "base_value": float(base_value),
    "max_additivity_error": max_additivity_error,
    "label_mapping": label_mapping,
}
with open(OUTPUT_DIR / "scz_shap_run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata_summary, f, ensure_ascii=False, indent=2)

print(f"\nتمام خروجی‌ها با موفقیت در پوشهٔ «{OUTPUT_DIR}» ذخیره شدند.")
