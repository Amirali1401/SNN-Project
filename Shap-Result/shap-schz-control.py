import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ۱. تنظیم مسیر پایه پروژه
BASE_DIR = Path(r"C:\Users\LENO\Desktop\deep learning")

OUTPUT_DIR = BASE_DIR / "Shap-Result" / "scz_shap_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND_SIZE = 50
EXPLAIN_SIZE = None     # برای تحلیل همه نمونه‌ها None بماند
NSAMPLES = "auto"
RANDOM_SEED = 42


def find_existing_file(candidates, file_label):
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"فایل {file_label} در مسیرهای زیر پیدا نشد:\n{tried}")


# ۲. پیدا کردن خودکار فایل‌های اسکیزوفرنی
data_candidates = [
    BASE_DIR / "Top - 100 - Schz .csv",
    BASE_DIR / "Top - 100 - Schz.csv",
    BASE_DIR / "schz_train.csv",
    BASE_DIR / "schizophrenia_artifacts" / "Top - 100 - Schz .csv",
    Path("Top - 100 - Schz .csv"),
]

genes_candidates = [
    BASE_DIR / "schizophrenia_artifacts" / "selected_genes.json",
    BASE_DIR / "schizophrenia_artifacts" / "feature_metadata.json",
    BASE_DIR / "scz_model_artifacts" / "selected_genes.json",
    BASE_DIR / "selected_genes.json",
]

imputer_candidates = [
    BASE_DIR / "schizophrenia_artifacts" / "imputer.pkl",
    BASE_DIR / "scz_model_artifacts" / "imputer.pkl",
    BASE_DIR / "imputer.pkl",
]

scaler_candidates = [
    BASE_DIR / "schizophrenia_artifacts" / "scaler.pkl",
    BASE_DIR / "scz_model_artifacts" / "scaler.pkl",
    BASE_DIR / "scaler.pkl",
]

model_candidates = [
    BASE_DIR / "schizophrenia_artifacts" / "snn_model.keras",
    BASE_DIR / "scz_model_artifacts" / "snn_model.keras",
    BASE_DIR / "snn_model.keras",
]

DATA_PATH = find_existing_file(data_candidates, "دیتاست اسکیزوفرنی")
GENES_PATH = find_existing_file(genes_candidates, "ژن‌های منتخب (selected_genes)")
IMPUTER_PATH = find_existing_file(imputer_candidates, "Imputer")
SCALER_PATH = find_existing_file(scaler_candidates, "Scaler")
MODEL_PATH = find_existing_file(model_candidates, "مدل شبکه عصبی (snn_model.keras)")

print(f"بارگذاری دیتاست از: {DATA_PATH}")
print(f"بارگذاری ژن‌ها از: {GENES_PATH}")
print(f"بارگذاری Imputer از: {IMPUTER_PATH}")
print(f"بارگذاری Scaler از: {SCALER_PATH}")
print(f"بارگذاری مدل از: {MODEL_PATH}")


# ۳. خواندن لیست ۱۰ ژن منتخب اسکیزوفرنی
def load_selected_genes(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(g).strip() for g in data]
    if isinstance(data, dict):
        for key in ("selected_genes", "selected_features", "genes", "features"):
            if key in data and isinstance(data[key], list):
                return [str(g).strip() for g in data[key]]
    raise ValueError(f"ساختار فایل ژن‌ها در {path} معتبر نیست.")


selected_genes = load_selected_genes(GENES_PATH)
print(f"تعداد ژن‌های منتخب بارگذاری‌شده: {len(selected_genes)}")
print(f"لیست ژن‌ها: {selected_genes}")


# ۴. بارگذاری ایمن Imputer، Scaler و مدل
imputer = joblib.load(IMPUTER_PATH)
scaler = joblib.load(SCALER_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# ۵. توابع مقاوم در برابر اختلاف نسخه scikit-learn
def safe_impute(imp, X_df: pd.DataFrame) -> np.ndarray:
    X_arr = X_df.to_numpy(dtype=np.float64, copy=True)

    # رفع باگ عدم تطابق نسخه (_fill_dtype و _fit_dtype)
    if not hasattr(imp, "_fill_dtype"):
        imp._fill_dtype = getattr(imp, "_fit_dtype", np.dtype("float64"))
    if not hasattr(imp, "_fit_dtype"):
        imp._fit_dtype = getattr(imp, "_fill_dtype", np.dtype("float64"))

    try:
        return np.asarray(imp.transform(X_df), dtype=np.float64)
    except Exception:
        # جایگزینی مستقیم مقادیر گمشده با آمار ذخیره شده در imputer.statistics_
        if hasattr(imp, "statistics_"):
            stats = np.asarray(imp.statistics_, dtype=np.float64)
            nan_mask = np.isnan(X_arr)
            if nan_mask.any():
                X_arr[nan_mask] = np.take(stats, np.where(nan_mask)[1])
            return X_arr
        return np.nan_to_num(X_arr, nan=np.nanmedian(X_arr, axis=0))


def safe_scale(scl, X_arr: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(scl.transform(X_arr), dtype=np.float64)
    except Exception:
        # استانداردسازی مستقیم با میانگین و انحراف معیار ذخیره شده در scaler
        if hasattr(scl, "mean_") and hasattr(scl, "scale_"):
            mean = np.asarray(scl.mean_, dtype=np.float64)
            scale = np.asarray(scl.scale_, dtype=np.float64)
            scale_safe = np.where(scale == 0, 1.0, scale)
            return (X_arr - mean) / scale_safe
        raise


def preprocess(df: pd.DataFrame) -> np.ndarray:
    missing_cols = [g for g in selected_genes if g not in df.columns]
    if missing_cols:
        raise KeyError(f"این ژن‌ها در ستون‌های فایل CSV یافت نشدند: {missing_cols}")

    X_df = df[selected_genes].apply(pd.to_numeric, errors="coerce")
    X_imputed = safe_impute(imputer, X_df)
    X_scaled = safe_scale(scaler, X_imputed)
    return X_scaled


# ۶. بارگذاری دیتاست و اجرای پیش‌پردازش
df_data = pd.read_csv(DATA_PATH)
df_data.columns = [str(c).strip() for c in df_data.columns]

X_all = preprocess(df_data)
print(f"ابعاد ماتریس داده پس از پیش‌پردازش: {X_all.shape}")


# ۷. انتخاب داده‌های Background و Explain برای SHAP
rng = np.random.default_rng(RANDOM_SEED)
n_total = X_all.shape[0]

bg_size = min(BACKGROUND_SIZE, n_total)
bg_indices = rng.choice(n_total, size=bg_size, replace=False)
X_background = X_all[bg_indices]

if EXPLAIN_SIZE is not None and EXPLAIN_SIZE < n_total:
    exp_indices = rng.choice(n_total, size=EXPLAIN_SIZE, replace=False)
    X_explain = X_all[exp_indices]
else:
    X_explain = X_all


# ۸. تابع پیش‌بینی مدل برای KernelExplainer
def model_predict(x_input):
    x_arr = np.asarray(x_input, dtype=np.float32)
    preds = model.predict(x_arr, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds.ravel()
    if preds.ndim == 2 and preds.shape[1] == 2:
        return preds[:, 1]
    return preds


# ۹. محاسبه مقادیر SHAP
print("در حال محاسبه مقادیر SHAP برای مدل اسکیزوفرنی...")
explainer = shap.KernelExplainer(model_predict, X_background)
shap_values = explainer.shap_values(X_explain, nsamples=NSAMPLES)

if isinstance(shap_values, list):
    shap_values = shap_values[-1]
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, -1]


# ۱۰. ساخت جدول رتبه‌بندی اهمیت ژن‌ها و ذخیره نتایج
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

importance_df = pd.DataFrame({
    "Gene_Probe": selected_genes,
    "Mean_Abs_SHAP": mean_abs_shap,
}).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

importance_df["Rank"] = np.arange(1, len(importance_df) + 1)

csv_out = OUTPUT_DIR / "scz_shap_feature_importance.csv"
raw_out = OUTPUT_DIR / "scz_shap_values_matrix.csv"

importance_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
pd.DataFrame(shap_values, columns=selected_genes).to_csv(raw_out, index=False, encoding="utf-8-sig")

print("\n۱۰ ژن برتر مدل اسکیزوفرنی بر اساس SHAP:")
print(importance_df.head(10).to_string())
print(f"\nفایل نتایج در مسیر زیر ذخیره شد:\n{csv_out}")
