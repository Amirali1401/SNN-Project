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

print("--- در حال آغاز پایپ‌لاین تحلیل SHAP برای مدل دوقطبی ---")

# =========================================================
# ۲. بارگذاری ژن‌های منتخب (Selected Genes)
# =========================================================
genes_candidates = [
    BASE_DIR / "selected_genes.json",
    BASE_DIR / "selected_genes-(2).json",
    BASE_DIR / "bipolar_artifacts" / "selected_genes.json",
    BASE_DIR / "selected_genes.csv",
    BASE_DIR / "bipolar_artifacts" / "selected_genes.csv",
]

selected_genes = None
for candidate in genes_candidates:
    if candidate.exists():
        print(f"بارگذاری ژن‌ها از: {candidate.name}")
        if candidate.suffix == ".json":
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
                selected_genes = data.get("selected_genes", data) if isinstance(data, dict) else data
        elif candidate.suffix == ".csv":
            df_g = pd.read_csv(candidate)
            col = "selected_genes" if "selected_genes" in df_g.columns else df_g.columns[0]
            selected_genes = df_g[col].dropna().tolist()
        break

if not selected_genes:
    raise FileNotFoundError("فایل ژن‌های منتخب (selected_genes) در مسیرهای مشخص‌شده یافت نشد.")

print(f"تعداد {len(selected_genes)} ژن با موفقیت لود شدند.")

# =========================================================
# ۳. بارگذاری مصنوعات (مدل، اسکیلر، ایمپیتر)
# =========================================================
# =========================================================
# ۳. بارگذاری مصنوعات (مدل، اسکیلر، ایمپیتر)
# =========================================================
model_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\bipolar_snn_model.keras"),
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\snn_model.keras"),
    BASE_DIR / "bipolar_artifacts" / "bipolar_snn_model.keras",
    BASE_DIR / "bipolar_artifacts" / "snn_model.keras",
    BASE_DIR / "snn_model.keras",
]

scaler_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\bipolar_scaler.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\scaler.pkl"),
    BASE_DIR / "scaler-(2).pkl",
    BASE_DIR / "scaler.pkl",
    BASE_DIR / "bipolar_artifacts" / "scaler.pkl",
]

imputer_candidates = [
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\bipolar_imputer.pkl"),
    Path(r"C:\Users\LENO\Desktop\deep learning\bipolar_artifacts\imputer.pkl"),
    BASE_DIR / "imputer-(2).pkl",
    BASE_DIR / "imputer.pkl",
    BASE_DIR / "bipolar_artifacts" / "imputer.pkl",
]

# پیدا کردن اولین مسیر معتبر
model_path = next((p for p in model_candidates if p.exists()), None)
scaler_path = next((p for p in scaler_candidates if p.exists()), None)
imputer_path = next((p for p in imputer_candidates if p.exists()), None)

if not model_path:
    raise FileNotFoundError(
        "فایل مدل کراس در هیچ‌یک از مسیرها پیدا نشد!\n"
        f"مسیرهای بررسی‌شده:\n" + "\n".join(str(p) for p in model_candidates)
    )

print(f"بارگذاری مدل از: {model_path}")
model = tf.keras.models.load_model(model_path)

scaler = joblib.load(scaler_path) if scaler_path else None
imputer = joblib.load(imputer_path) if imputer_path else None

if scaler_path:
    print(f"بارگذاری اسکیلر از: {scaler_path.name}")
if imputer_path:
    print(f"بارگذاری ایمپیتر از: {imputer_path.name}")

model_path = next((p for p in model_candidates if p.exists()), None)
scaler_path = next((p for p in scaler_candidates if p.exists()), None)
imputer_path = next((p for p in imputer_candidates if p.exists()), None)

if not model_path:
    raise FileNotFoundError("فایل مدل snn_model.keras یافت نشد.")

print(f"بارگذاری مدل از: {model_path.name}")
model = tf.keras.models.load_model(model_path)

scaler = joblib.load(scaler_path) if scaler_path else None
imputer = joblib.load(imputer_path) if imputer_path else None

# =======================================================
# =========================================================
# ۴. آماده‌سازی داده‌های تست و پس‌زمینه (نسخه مقاوم و اصلاح‌شده)
# =========================================================
if not TEST_PATH.exists():
    raise FileNotFoundError(f"فایل تست در مسیر {TEST_PATH} یافت نشد.")

df_test = pd.read_csv(TEST_PATH)

# استخراج ستون‌های داده و حذف ستون‌های غیرضروری احتمالی (مثل sample_id یا Group)
for col_to_drop in ["sample_id", "Group", "ID_REF"]:
    if col_to_drop in df_test.columns and col_to_drop not in selected_genes:
        # اگر ستون هدف در داده تست هست اما جزء ژن‌ها نیست، موقتاً نگه میداریم یا جدا میکنیم
        pass

# تضمین اینکه دقیقاً همان ژن‌های منتخب مدل (معمولاً ۳۸ ژن) با همان ترتیب استخراج شوند
missing_in_test = [g for g in selected_genes if g not in df_test.columns]
if missing_in_test:
    print(f"هشدار: این ژن‌ها در فایل تست نبودند و مقدار صفر در نظر گرفته می‌شوند: {missing_in_test}")
    for g in missing_in_test:
        df_test[g] = 0.0

# فیلتر کردن دقیق دی‌اف تست روی selected_genes به همان ترتیبی که مدل آموزش دیده
X_test_raw = df_test[selected_genes].copy()
X_test_raw = X_test_raw.apply(pd.to_numeric, errors="coerce").fillna(0)

# اعمال ایمپیتر و اسکیلر با بررسی ابعاد
try:
    if imputer:
        X_test_imp = imputer.transform(X_test_raw)
    else:
        X_test_imp = X_test_raw.values

    if scaler:
        X_processed = scaler.transform(X_test_imp)
    else:
        X_processed = X_test_imp
except Exception as e:
    print(f"خطا در اعمال Scaler یا Imputer: {e}")
    # روش جایگزین اضطراری اگر ابعاد اسکیلر با ستون‌ها نخواند
    X_processed = X_test_raw.values

X_processed = np.asarray(X_processed, dtype=np.float32)

# آماده‌سازی داده پس‌زمینه (Background)
if BACKGROUND_PATH.exists():
    print(f"در حال ساخت دیتای پس‌زمینه از: {BACKGROUND_PATH.name}")
    df_bg = pd.read_csv(BACKGROUND_PATH)
    
    # اطمینان از وجود ستون‌های ژن در پس‌زمینه
    bg_cols = [g for g in selected_genes if g in df_bg.columns]
    X_bg_raw = df_bg[bg_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    
    # اگر تعداد ستون‌ها کامل بود از ایمپیتر اصلی استفاده کن
    try:
        X_bg_imp = imputer.transform(X_bg_raw) if imputer else X_bg_raw.values
        X_bg_scaled = scaler.transform(X_bg_imp) if scaler else X_bg_imp
    except:
        X_bg_scaled = X_bg_raw.values

    sample_size = min(BACKGROUND_SIZE, len(X_bg_scaled))
    background_data = np.asarray(X_bg_scaled[:sample_size], dtype=np.float32)
else:
    sample_size = min(BACKGROUND_SIZE, len(X_processed))
    background_data = np.asarray(X_processed[:sample_size], dtype=np.float32)

print(f"ابعاد داده‌های پردازش‌شده تست: {X_processed.shape}")
print(f"ابعاد ماتریس پس‌زمینه (Background Shape): {background_data.shape}")

# تعیین متغیر valid_genes برای استفاده در مراحل بعدی
valid_genes = selected_genes
