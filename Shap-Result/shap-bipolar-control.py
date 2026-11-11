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
# ۱. تنظیمات مسیرها (با استفاده از شیء Path و رشته خام r)
# =========================================================
BASE_DIR = Path(r"C:\Users\LENO\Desktop\deep learning")
TEST_PATH = BASE_DIR / "Top - 100 - Bipolar.csv"
BACKGROUND_PATH = BASE_DIR / "snn_train_input.csv"
OUTPUT_DIR = BASE_DIR / "bipolar_shap_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BACKGROUND_SIZE = 30
TARGET_CLASS_INDEX = 1

np.random.seed(RANDOM_SEED)
print("--- در حال آغاز پایپ‌لاین تحلیل SHAP برای مدل دوقطبی ---")

# =========================================================
# ۲. بارگذاری هوشمند ژن‌های منتخب (Selected Genes)
# =========================================================
# اولویت با فایل‌های اختصاصی دوقطبی و فایل CSV دارای ۳۸ ژن است
genes_candidates = [
    BASE_DIR / "bipolar_artifacts" / "selected_genes.csv",
    BASE_DIR / "bipolar_artifacts" / "selected_genes.json",
    BASE_DIR / "selected_genes.csv",
    BASE_DIR / "selected_genes-(2).json",
    BASE_DIR / "selected_genes.json",
]

def extract_valid_genes_from_df(df_g: pd.DataFrame) -> list:
    """استخراج هوشمند ستون نام ژن‌ها و نادیده گرفتن ستون شماره ردیف (rank)"""
    # ۱. بررسی نام‌های استاندارد ستون ژن
    for col_name in ["gene", "Gene", "selected_genes", "probe", "Probe", "ID_REF", "feature"]:
        if col_name in df_g.columns:
            return df_g[col_name].dropna().astype(str).str.strip().tolist()
    
    # ۲. اگر نام ستون استاندارد نبود، ستونی را انتخاب کن که شامل '_at' باشد (فرمت پروب‌های Affymetrix)
    for col in df_g.columns:
        sample_vals = df_g[col].dropna().astype(str).tolist()
        if any("_at" in val for val in sample_vals[:5]):
            return [v.strip() for v in sample_vals]
            
    # ۳. در غیر این صورت، اگر بیش از یک ستون داریم و ستون اول عددی است، ستون دوم را بردار
    if len(df_g.columns) > 1 and pd.api.types.is_numeric_dtype(df_g.iloc[:, 0]):
        return df_g.iloc[:, 1].dropna().astype(str).str.strip().tolist()
        
    return df_g.iloc[:, 0].dropna().astype(str).str.strip().tolist()

selected_genes = None
for candidate in genes_candidates:
    if candidate.exists():
        if candidate.suffix == ".csv":
            df_g = pd.read_csv(candidate)
            genes_list = extract_valid_genes_from_df(df_g)
        elif candidate.suffix == ".json":
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    genes_list = data.get("selected_genes", data.get("genes", list(data.values())[0]))
                else:
                    genes_list = data
                genes_list = [str(g).strip() for g in genes_list]
        
        # اطمینان از اینکه فایل ۱۰ ژنی اسکیزوفرنی به اشتباه برای دوقطبی (۳۸ ژنی) لود نشود
        if len(genes_list) == 38 or candidate == genes_candidates[-1]:
            selected_genes = genes_list
            print(f"بارگذاری ژن‌ها از: {candidate.name}")
            break

if not selected_genes:
    raise FileNotFoundError("فایل ژن‌های منتخب (selected_genes) در مسیرهای مشخص‌شده یافت نشد.")

print(f"تعداد {len(selected_genes)} ژن با موفقیت لود شدند.")
print(f"نمونه ۵ ژن اول: {selected_genes[:5]}")

# =========================================================
# ۳. بارگذاری مصنوعات (مدل، اسکیلر، ایمپیتر)
# =========================================================
model_candidates = [
    BASE_DIR / "bipolar_artifacts" / "bipolar_snn_model.keras",
    BASE_DIR / "bipolar_artifacts" / "snn_model.keras",
    BASE_DIR / "bipolar_snn_model.keras",
    BASE_DIR / "snn_model.keras",
]

scaler_candidates = [
    BASE_DIR / "bipolar_artifacts" / "bipolar_scaler.pkl",
    BASE_DIR / "bipolar_artifacts" / "scaler.pkl",
    BASE_DIR / "bipolar_scaler.pkl",
    BASE_DIR / "scaler-(2).pkl",
    BASE_DIR / "scaler.pkl",
]

imputer_candidates = [
    BASE_DIR / "bipolar_artifacts" / "bipolar_imputer.pkl",
    BASE_DIR / "bipolar_artifacts" / "imputer.pkl",
    BASE_DIR / "bipolar_imputer.pkl",
    BASE_DIR / "imputer-(2).pkl",
    BASE_DIR / "imputer.pkl",
]

model_path = next((p for p in model_candidates if p.exists()), None)
scaler_path = next((p for p in scaler_candidates if p.exists()), None)
imputer_path = next((p for p in imputer_candidates if p.exists()), None)

if not model_path:
    raise FileNotFoundError(
        "فایل مدل کراس در هیچ‌یک از مسیرها پیدا نشد!\n"
        f"مسیرهای بررسی‌شده:\n" + "\n".join(str(p) for p in model_candidates)
    )

print(f"بارگذاری مدل از: {model_path.name}")
model = tf.keras.models.load_model(model_path)

scaler = joblib.load(scaler_path) if scaler_path else None
imputer = joblib.load(imputer_path) if imputer_path else None

if scaler_path:
    print(f"بارگذاری اسکیلر از: {scaler_path.name}")
if imputer_path:
    print(f"بارگذاری ایمپیتر از: {imputer_path.name}")

# --- توابع ایمن برای جلوگیری از خطای نسخه scikit-learn (مثل _fill_dtype) ---
def safe_impute(X_df: pd.DataFrame, imp) -> np.ndarray:
    X_arr = np.asarray(X_df, dtype=np.float64)
    if imp is None:
        return np.nan_to_num(X_arr, nan=0.0)
    try:
        if not hasattr(imp, "_fill_dtype"):
            imp._fill_dtype = np.dtype("float64")
        return imp.transform(X_arr)
    except Exception:
        if hasattr(imp, "statistics_"):
            stats = np.asarray(imp.statistics_, dtype=np.float64)
            inds = np.where(np.isnan(X_arr))
            X_arr[inds] = np.take(stats, inds[1])
            return X_arr
        return np.nan_to_num(X_arr, nan=0.0)

def safe_scale(X_arr: np.ndarray, scl) -> np.ndarray:
    X_arr = np.asarray(X_arr, dtype=np.float64)
    if scl is None:
        return X_arr
    try:
        return scl.transform(X_arr)
    except Exception:
        if hasattr(scl, "mean_") and hasattr(scl, "scale_"):
            return (X_arr - np.asarray(scl.mean_)) / np.where(np.asarray(scl.scale_) == 0, 1.0, np.asarray(scl.scale_))
        return X_arr

# =========================================================
# ۴. آماده‌سازی داده‌های تست و پس‌زمینه (نسخه مقاوم و اصلاح‌شده)
# =========================================================
if not TEST_PATH.exists():
    raise FileNotFoundError(f"فایل تست در مسیر {TEST_PATH} یافت نشد.")

df_test = pd.read_csv(TEST_PATH)
df_test.columns = df_test.columns.astype(str).str.strip()

# بررسی وجود ژن‌ها در فایل تست
missing_in_test = [g for g in selected_genes if g not in df_test.columns]
if missing_in_test:
    print(f"هشدار: این ژن‌ها در فایل تست نبودند و مقدار صفر در نظر گرفته می‌شوند: {missing_in_test}")
    for g in missing_in_test:
        df_test[g] = 0.0
else:
    print("تأیید: تمام ژن‌های منتخب (۱۰۰٪) در فایل تست پیدا شدند.")

# فیلتر کردن دقیق دیتافریم تست روی selected_genes با همان ترتیب آموزشی
X_test_raw = df_test[selected_genes].apply(pd.to_numeric, errors="coerce")

# اعمال ایمپیتر و اسکیلر به صورت ایمن
X_test_imp = safe_impute(X_test_raw, imputer)
X_processed = safe_scale(X_test_imp, scaler)
X_processed = np.asarray(X_processed, dtype=np.float32)

# آماده‌سازی داده پس‌زمینه (Background)
if BACKGROUND_PATH.exists():
    print(f"در حال ساخت دیتای پس‌زمینه از: {BACKGROUND_PATH.name}")
    df_bg = pd.read_csv(BACKGROUND_PATH)
    df_bg.columns = df_bg.columns.astype(str).str.strip()
    
    for g in selected_genes:
        if g not in df_bg.columns:
            df_bg[g] = 0.0
            
    X_bg_raw = df_bg[selected_genes].apply(pd.to_numeric, errors="coerce")
    X_bg_imp = safe_impute(X_bg_raw, imputer)
    X_bg_scaled = safe_scale(X_bg_imp, scaler)
    
    sample_size = min(BACKGROUND_SIZE, len(X_bg_scaled))
    background_data = np.asarray(X_bg_scaled[:sample_size], dtype=np.float32)
else:
    sample_size = min(BACKGROUND_SIZE, len(X_processed))
    background_data = np.asarray(X_processed[:sample_size], dtype=np.float32)

print(f"ابعاد داده‌های پردازش‌شده تست: {X_processed.shape}")
print(f"ابعاد ماتریس پس‌زمینه (Background Shape): {background_data.shape}")

valid_genes = selected_genes

# =========================================================
# ۵. تعریف تابع پیش‌بینی و اجرای SHAP KernelExplainer
# =========================================================
def predict_fn(data):
    data_arr = np.asarray(data, dtype=np.float32)
    preds = model.predict(data_arr, verbose=0)
    preds = np.asarray(preds)
    # اگر خروجی مدل دو ستون احتمال (کلاس ۰ و ۱) داشت، ستون کلاس هدف را برگردان
    if preds.ndim == 2 and preds.shape[1] > 1:
        return preds[:, TARGET_CLASS_INDEX]
    return preds.reshape(-1)

print("در حال محاسبه مقادیر SHAP (لطفاً کمی صبر کنید)...")
explainer = shap.KernelExplainer(predict_fn, background_data)
shap_values = explainer.shap_values(X_processed, nsamples="auto")

# یکسان‌سازی ابعاد آرایه خروجی SHAP در نسخه‌های مختلف کتابخانه
if isinstance(shap_values, list):
    shap_matrix = np.asarray(shap_values[TARGET_CLASS_INDEX if len(shap_values) > 1 else 0])
else:
    shap_matrix = np.asarray(shap_values)
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, TARGET_CLASS_INDEX if shap_matrix.shape[2] > 1 else 0]

# =========================================================
# ۶. محاسبه اهمیت ویژگی‌ها، ذخیره جدول و رسم نمودارها
# =========================================================
mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

importance_df = pd.DataFrame({
    "Gene_Probe": valid_genes,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

importance_df["Rank"] = np.arange(1, len(importance_df) + 1)

csv_out_path = OUTPUT_DIR / "bipolar_shap_feature_importance.csv"
importance_df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

print("\n۱۰ ژن برتر مدل دوقطبی بر اساس SHAP:")
print(importance_df.head(10).to_string(index=True))
print(f"\nجدول کامل اهمیت ژن‌ها در مسیر زیر ذخیره شد:\n{csv_out_path}")

# رسم نمودار Beeswarm Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_matrix,
    features=X_processed,
    feature_names=valid_genes,
    max_display=20,
    show=False
)
plt.title("SHAP Summary Plot - Bipolar Disorder (Top 20 Genes)", fontsize=13, pad=15)
plt.tight_layout()
summary_plot_path = OUTPUT_DIR / "bipolar_shap_summary_beeswarm.png"
plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# رسم نمودار میله‌ای (Bar Plot)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_matrix,
    features=X_processed,
    feature_names=valid_genes,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.title("SHAP Feature Importance (Bar) - Bipolar")