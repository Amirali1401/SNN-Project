import json
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

# نادیده گرفتن هشدارهای غیرضروری
warnings.filterwarnings("ignore")

# =========================================================
# ۱. تنظیمات مسیرها (به صورت آدرس کامل و مستقیم ویندوز)
# =========================================================
OUTPUT_DIR = Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\Schizophrenia")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND_SIZE = 50
EXPLAIN_SIZE = None     # برای تحلیل همه نمونه‌ها None بماند
NSAMPLES = "auto"
RANDOM_SEED = 42
TARGET_CLASS_INDEX = 1  # کلاس ۱ معادل اسکیزوفرنی

def get_existing_path(candidates_list):
    for item in candidates_list:
        p = Path(item)
        if p.exists():
            return p
    return None

# لیست مسیرهای احتمالی فایل داده اسکیزوفرنی (توجه: فاصله قبل از .csv در نام فایل لحاظ شده است)
data_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\Top - 100 - Schz .csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\Top - 100 - Schz.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\scz_model_artifacts\Top - 100 - Schz .csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\schizophrenia_artifacts\Top - 100 - Schz .csv"),
]

# لیست مسیرهای احتمالی فایل مرجع (آموزش) در صورت وجود
background_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\scz_train_input.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\scz_model_artifacts\scz_train_input.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\schizophrenia_artifacts\snn_train_input.csv"),
]

DATA_PATH = get_existing_path(data_candidates)
BACKGROUND_PATH = get_existing_path(background_candidates)

if DATA_PATH is None:
    raise FileNotFoundError("فایل داده اسکیزوفرنی (Top - 100 - Schz .csv) پیدا نشد! آدرس را در data_candidates بررسی کنید.")

print(f"فایل داده تست اسکیزوفرنی: {DATA_PATH}")

# =========================================================
# ۲. بارگذاری هوشمند ژن‌های منتخب (جلوگیری از خواندن ستون rank)
# =========================================================
genes_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\scz_model_artifacts\selected_genes.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\schizophrenia_artifacts\selected_genes.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\selected_genes.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\scz_model_artifacts\selected_genes.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\schizophrenia_artifacts\selected_genes.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\selected_genes.csv"),
]

selected_genes = None
for candidate in genes_candidates:
    candidate = Path(candidate)
    if candidate.exists():
        if candidate.suffix == ".json":
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    selected_genes = (
                        data.get("selected_genes")
                        or data.get("selected_features")
                        or list(data.values())[0]
                    )
                elif isinstance(data, list):
                    selected_genes = data

        elif candidate.suffix == ".csv":
            df_g = pd.read_csv(candidate)
            for col_name in ["gene", "feature", "selected_genes", "probe"]:
                if col_name in df_g.columns:
                    selected_genes = df_g[col_name].dropna().astype(str).tolist()
                    break
            if not selected_genes:
                for c in df_g.columns:
                    if not pd.to_numeric(df_g[c], errors="coerce").notnull().all():
                        selected_genes = df_g[c].dropna().astype(str).tolist()
                        break

        if selected_genes and not str(selected_genes[0]).isdigit():
            print(f"ژن‌های منتخب از '{candidate}' بارگذاری شدند.")
            break

if not selected_genes or str(selected_genes[0]).isdigit():
    raise ValueError("فایل ژن‌های منتخب اسکیزوفرنی پیدا نشد یا اعداد رتبه به جای نام ژن خوانده شده‌اند.")

print(f"تعداد ژن‌های انتخاب‌شده برای اسکیزوفرنی: {len(selected_genes)} | نمونه: {selected_genes[:3]}")

# =========================================================
# ۳. بارگذاری مدل، Imputer، Scaler و Label Mapping
# =========================================================
imputer_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\scz_model_artifacts\imputer.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\schizophrenia_artifacts\imputer.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\imputer.pkl"),
]

scaler_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\scz_model_artifacts\scaler.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\schizophrenia_artifacts\scaler.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\scaler.pkl"),
]

model_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\scz_model_artifacts\snn_model.keras"),
    Path(r"C:\Users\LENO\Desktop\deep learning\schizophrenia_artifacts\snn_model.keras"),
    Path(r"C:\Users\LENO\Desktop\deep learning\snn_model.keras"),
]

label_map_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\scz_model_artifacts\label_mapping.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\schizophrenia_artifacts\label_mapping.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\label_mapping.json"),
]

imputer_path = get_existing_path(imputer_candidates)
scaler_path = get_existing_path(scaler_candidates)
model_path = get_existing_path(model_candidates)
label_map_path = get_existing_path(label_map_candidates)

if model_path is None:
    raise FileNotFoundError("فایل مدل اسکیزوفرنی (snn_model.keras) پیدا نشد!")

print(f"بارگذاری مدل از: {model_path}")
model = tf.keras.models.load_model(model_path)

imputer = joblib.load(imputer_path) if imputer_path else None
scaler = joblib.load(scaler_path) if scaler_path else None

if imputer_path:
    print(f"بارگذاری Imputer از: {imputer_path}")
if scaler_path:
    print(f"بارگذاری Scaler از: {scaler_path}")

label_mapping = {}
if label_map_path is not None:
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

# =========================================================
# ۴. پیش‌پردازش داده‌ها (تبدیل به آرایه NumPy برای جلوگیری از خطای نام ستون‌ها)
# =========================================================
def preprocess(df: pd.DataFrame) -> np.ndarray:
    df_copy = df.copy()
    missing_genes = [g for g in selected_genes if g not in df_copy.columns]
    if missing_genes:
        print(f"هشدار: این ژن‌ها در فایل داده نبودند و با صفر جایگزین می‌شوند: {missing_genes}")
        for g in missing_genes:
            df_copy[g] = 0.0

    # استفاده از .values برای ارسال آرایه خام به imputer و scaler جهت جلوگیری از ValueError در sklearn
    X = df_copy[selected_genes].apply(pd.to_numeric, errors="coerce").values

    if imputer is not None:
        X = imputer.transform(X)
    else:
        X = np.nan_to_num(X, nan=0.0)

    if scaler is not None:
        X = scaler.transform(X)

    return np.asarray(X, dtype=np.float32)

df_data = pd.read_csv(DATA_PATH)
X_all = preprocess(df_data)

rng = np.random.default_rng(RANDOM_SEED)

if BACKGROUND_PATH is not None and BACKGROUND_PATH.exists():
    print(f"بارگذاری داده پس‌زمینه از: {BACKGROUND_PATH}")
    df_bg = pd.read_csv(BACKGROUND_PATH)
    X_bg_pool = preprocess(df_bg)
else:
    print("فایل مرجع جداگانه معرفی نشده است؛ داده مرجع به صورت تصادفی از فایل فعلی انتخاب می‌شود.")
    X_bg_pool = X_all

bg_size = min(BACKGROUND_SIZE, len(X_bg_pool))
bg_indices = rng.choice(len(X_bg_pool), size=bg_size, replace=False)
X_background = np.asarray(X_bg_pool[bg_indices], dtype=np.float32)

if EXPLAIN_SIZE is not None and EXPLAIN_SIZE < len(X_all):
    explain_indices = rng.choice(len(X_all), size=EXPLAIN_SIZE, replace=False)
    X_explain = np.asarray(X_all[explain_indices], dtype=np.float32)
else:
    X_explain = np.asarray(X_all, dtype=np.float32)

print(f"ابعاد داده‌های پردازش‌شده تست: {X_explain.shape}")
print(f"ابعاد ماتریس پس‌زمینه (Background Shape): {X_background.shape}")

# =========================================================
# ۵. تعریف تابع پیش‌بینی و اجرای تحلیل SHAP
# =========================================================
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

print("در حال محاسبه مقادیر SHAP برای مدل اسکیزوفرنی (لطفاً کمی صبر کنید)...")
# ارسال مستقیم آرایه ۲بعدی X_background برای جلوگیری از خطای Independent masker
explainer = shap.KernelExplainer(predict_target, X_background)
shap_values_raw = explainer.shap_values(X_explain, nsamples=NSAMPLES, l1_reg=0.0)

if isinstance(shap_values_raw, list):
    shap_values = np.asarray(shap_values_raw[-1], dtype=np.float64)
else:
    shap_values = np.asarray(shap_values_raw, dtype=np.float64)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

# =========================================================
# ۶. رتبه‌بندی ژن‌ها، کنترل خطای بازسازی و ذخیره فایل‌ها
# =========================================================
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
mean_signed_shap = np.mean(shap_values, axis=0)

ranking_df = pd.DataFrame({
    "gene": selected_genes,
    "mean_abs_shap": mean_abs_shap,
    "mean_signed_shap": mean_signed_shap,
}).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)
ranking_df = ranking_df[["rank", "gene", "mean_abs_shap", "mean_signed_shap"]]

base_value = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
reconstructed = base_value + np.sum(shap_values, axis=1)
actual_preds = predict_target(X_explain)
max_additivity_error = float(np.max(np.abs(reconstructed - actual_preds)))

print(f"\nبیشترین خطای بازسازی خروجی مدل از SHAP: {max_additivity_error:.6e}\n")
print("رتبه‌بندی ژن‌های برتر در مدل اسکیزوفرنی:")
print(ranking_df.to_string(index=False))

# آدرس‌های مستقیم خروجی‌ها
importance_csv_path = Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\Schizophrenia\scz_shap_feature_importance.csv")
samples_csv_path = Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\Schizophrenia\scz_shap_values_per_sample.csv")
metadata_json_path = Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\Schizophrenia\scz_shap_run_metadata.json")
bar_plot_path = Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\Schizophrenia\scz_shap_bar_plot.png")
beeswarm_plot_path = Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\Schizophrenia\scz_shap_beeswarm.png")

ranking_df.to_csv(importance_csv_path, index=False)

shap_samples_df = pd.DataFrame(shap_values, columns=selected_genes)
if "ID_REF" in df_data.columns and len(shap_samples_df) == len(df_data):
    shap_samples_df.insert(0, "ID_REF", df_data["ID_REF"].values)
if "Group" in df_data.columns and len(shap_samples_df) == len(df_data):
    shap_samples_df.insert(1 if "ID_REF" in shap_samples_df.columns else 0, "Group", df_data["Group"].values)

shap_samples_df.to_csv(samples_csv_path, index=False)

metadata_summary = {
    "task": "schizophrenia_binary_classification",
    "num_genes": len(selected_genes),
    "num_explained_samples": int(len(X_explain)),
    "background_size": int(len(X_background)),
    "base_value": base_value,
    "max_additivity_error": max_additivity_error,
    "label_mapping": label_mapping,
}
with open(metadata_json_path, "w", encoding="utf-8") as f:
    json.dump(metadata_summary, f, ensure_ascii=False, indent=2)

# رسم نمودار Bar
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_explain, feature_names=selected_genes, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - Schizophrenia SNN Model")
plt.tight_layout()
plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# رسم نمودار Beeswarm
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_explain, feature_names=selected_genes, show=False)
plt.title("SHAP Summary Plot (Beeswarm) - Schizophrenia SNN Model")
plt.tight_layout()
plt.savefig(beeswarm_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nتمام فایل‌ها و نمودارها با موفقیت در پوشه زیر ذخیره شدند:\n{OUTPUT_DIR}")
