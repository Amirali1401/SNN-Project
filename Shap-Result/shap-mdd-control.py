import json
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

warnings.filterwarnings("ignore")

# =========================================================
# ۱. تنظیمات مسیرها (به صورت آدرس کامل و مستقیم ویندوز)
# =========================================================
OUTPUT_DIR = Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\MDD")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# لیست مسیرهای احتمالی فایل تست
test_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\GSE98793_MDD_labeled.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\GSE98793_MDD_100_probes.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_test.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_train.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\merged_MDD_Bipolar_100probes_labeled.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\mdd_train.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\mdd_train.csv"),
]

# لیست مسیرهای احتمالی فایل آموزش (پس‌زمینه SHAP)
background_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_train.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\mdd_train.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\mdd_train.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\snn_train_input.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\GSE98793_MDD_labeled.csv"),
]

def get_existing_path(candidates_list):
    for item in candidates_list:
        p = Path(item)  # تضمین تبدیل به Path حتی اگر رشته ساده وارد شده باشد
        if p.exists():
            return p
    return None

TEST_PATH = get_existing_path(test_candidates)
BACKGROUND_PATH = get_existing_path(background_candidates)

if TEST_PATH is None:
    raise FileNotFoundError("فایل داده تست MDD پیدا نشد! آدرس فایل تست خود را در لیست test_candidates قرار دهید.")

print(f"فایل داده تست: {TEST_PATH}")
print(f"فایل داده پس‌زمینه: {BACKGROUND_PATH if BACKGROUND_PATH else TEST_PATH}")

# =========================================================
# ۲. بارگذاری ژن‌های منتخب (با آدرس مستقیم)
# =========================================================
genes_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\selected_genes.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\selected_genes.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_artifacts\selected_genes.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\selected_genes.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\selected_features.json"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\selected_genes.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\selected_genes.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\selected_genes.csv"),
    Path(r"C:\Users\LENO\Desktop\deep learning\selected_features.csv"),
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
            print(f"ژن‌ها با موفقیت از '{candidate}' بارگذاری شدند.")
            break

if not selected_genes or str(selected_genes[0]).isdigit():
    raise ValueError("فایل ژن‌های منتخب پیدا نشد یا نام پروب‌ها به درستی استخراج نشدند.")

print(f"تعداد {len(selected_genes)} ژن معتبر لود شد. نمونه ۳ پروب اول: {selected_genes[:3]}")

# =========================================================
# ۳. بارگذاری مدل، Imputer و Scaler (با آدرس مستقیم)
# =========================================================
imputer_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\mdd_imputer.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\imputer.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\mdd_imputer.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\imputer.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\imputer.pkl"),
]

scaler_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\mdd_scaler.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\scaler.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\mdd_scaler.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\scaler.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\scaler.pkl"),
]

model_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\mdd_snn_model.keras"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts_small\snn_model.keras"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\mdd_snn_model.keras"),
    Path(r"C:\Users\LENO\Desktop\deep learning\mdd_model_artifacts\snn_model.keras"),
    Path(r"C:\Users\LENO\Desktop\deep learning\snn_model.keras"),
]

imputer_path = get_existing_path(imputer_candidates)
scaler_path = get_existing_path(scaler_candidates)
model_path = get_existing_path(model_candidates)

if model_path is None:
    raise FileNotFoundError("فایل مدل (.keras) پیدا نشد! آدرس دقیق آن را در لیست model_candidates بگذارید.")

print(f"بارگذاری مدل از: {model_path}")
model = tf.keras.models.load_model(model_path)

imputer = joblib.load(imputer_path) if imputer_path else None
scaler = joblib.load(scaler_path) if scaler_path else None

if imputer_path:
    print(f"بارگذاری Imputer از: {imputer_path}")
if scaler_path:
    print(f"بارگذاری Scaler از: {scaler_path}")

# =========================================================
# ۴. آماده‌سازی داده‌ها و اجرای SHAP
# =========================================================
def prepare_data(csv_path, genes_list, imputer_obj=None, scaler_obj=None):
    df = pd.read_csv(csv_path)
    missing_genes = [g for g in genes_list if g not in df.columns]
    if missing_genes:
        print(f"هشدار: {len(missing_genes)} ژن در فایل نبودند و صفر در نظر گرفته می‌شوند: {missing_genes[:5]}")
        for g in missing_genes:
            df[g] = 0.0

    X = df[genes_list].apply(pd.to_numeric, errors="coerce").values

    if imputer_obj is not None:
        X = imputer_obj.transform(X)
    else:
        X = np.nan_to_num(X, nan=0.0)

    if scaler_obj is not None:
        X = scaler_obj.transform(X)

    return np.asarray(X, dtype=np.float32)

X_test_processed = prepare_data(TEST_PATH, selected_genes, imputer, scaler)
bg_source = BACKGROUND_PATH if BACKGROUND_PATH is not None else TEST_PATH
X_bg_full = prepare_data(bg_source, selected_genes, imputer, scaler)

np.random.seed(42)
bg_size = min(30, len(X_bg_full))
bg_indices = np.random.choice(len(X_bg_full), size=bg_size, replace=False)
background_data = np.asarray(X_bg_full[bg_indices], dtype=np.float32)

print(f"ابعاد داده‌های پردازش‌شده تست: {X_test_processed.shape}")
print(f"ابعاد ماتریس پس‌زمینه (Background Shape): {background_data.shape}")

def model_predict(data_array):
    data_2d = np.asarray(data_array, dtype=np.float32)
    preds = model.predict(data_2d, verbose=0)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds.ravel()
    elif preds.ndim == 2 and preds.shape[1] > 1:
        return preds[:, 1]
    return preds.ravel()

print("در حال محاسبه مقادیر SHAP برای مدل MDD (لطفاً کمی صبر کنید)...")
explainer = shap.KernelExplainer(model_predict, background_data)
shap_values = explainer.shap_values(X_test_processed, nsamples="auto")

if isinstance(shap_values, list):
    shap_matrix = np.asarray(shap_values[-1])
else:
    shap_matrix = np.asarray(shap_values)
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, -1]

# =========================================================
# ۵. ذخیره‌سازی نتایج و نمودارها
# =========================================================
mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
importance_df = pd.DataFrame({
    "Gene_Probe": selected_genes,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

importance_df["Rank"] = importance_df.index + 1
csv_out = Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\MDD\mdd_shap_feature_importance.csv")
importance_df.to_csv(csv_out, index=False)
print(f"\nجدول رتبه‌بندی ژن‌های MDD ذخیره شد: {csv_out}")
print(importance_df.head(10))

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_matrix, X_test_processed, feature_names=selected_genes, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - MDD SNN Model")
plt.tight_layout()
plt.savefig(Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\MDD\mdd_shap_bar_plot.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_matrix, X_test_processed, feature_names=selected_genes, show=False)
plt.title("SHAP Summary Plot (Beeswarm) - MDD SNN Model")
plt.tight_layout()
plt.savefig(Path(r"C:\Users\LENO\Desktop\deep learning\Shap-Result\MDD\mdd_shap_beeswarm.png"), dpi=300, bbox_inches="tight")
plt.close()

print("نمودارها با موفقیت ذخیره شدند.")
