from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# تنظیمات
# ============================================================

INPUT_FILE = Path("")

TRAIN_FILE = Path("combined_train.csv")
TEST_FILE = Path("combined_test.csv")

# نسبت دادهٔ تست
TEST_SIZE = 0.20

# برای تکرارپذیر بودن نتیجه
RANDOM_STATE = 42


# ============================================================
# خواندن فایل
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"فایل ورودی پیدا نشد:\n{INPUT_FILE.resolve()}\n"
        "نام فایل را در متغیر INPUT_FILE اصلاح کن."
    )

df = pd.read_csv(INPUT_FILE)

print(f"فایل ورودی: {INPUT_FILE}")
print(f"ابعاد داده: {df.shape}")


# ============================================================
# بررسی ساختار داده
# ============================================================

if "Group" not in df.columns:
    raise ValueError(
        "ستون Group در فایل پیدا نشد. "
        "نام ستون برچسب باید دقیقاً Group باشد."
    )

if "ID_REF" not in df.columns:
    print("هشدار: ستون ID_REF وجود ندارد؛ ادامهٔ پردازش انجام می‌شود.")

if df.empty:
    raise ValueError("فایل داده خالی است.")

if df["Group"].isna().any():
    raise ValueError("برخی نمونه‌ها فاقد برچسب Group هستند.")

if df["Group"].nunique() != 2:
    raise ValueError(
        f"تعداد کلاس‌ها باید دقیقاً ۲ باشد، "
        f"اما {df['Group'].nunique()} کلاس پیدا شد:\n"
        f"{df['Group'].unique()}"
    )

# حذف فاصله‌های اضافی از برچسب‌ها
df["Group"] = df["Group"].astype(str).str.strip()

class_counts = df["Group"].value_counts()

print("\nتوزیع کل کلاس‌ها:")
print(class_counts)

if class_counts.min() < 2:
    raise ValueError(
        "حداقل یکی از کلاس‌ها کمتر از ۲ نمونه دارد و تقسیم stratified ممکن نیست."
    )


# ============================================================
# تقسیم Stratified
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["Group"],
    shuffle=True
)

# مرتب‌سازی مجدد index
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# ذخیرهٔ خروجی‌ها
# ============================================================

train_df.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")
test_df.to_csv(TEST_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# گزارش نهایی
# ============================================================

print("\nتقسیم با موفقیت انجام شد.")

print(f"\nفایل Train:")
print(f"  نام: {TRAIN_FILE.resolve()}")
print(f"  تعداد نمونه: {len(train_df)}")
print(f"  ابعاد: {train_df.shape}")

print("\nتوزیع کلاس‌ها در Train:")
print(train_df["Group"].value_counts())

print(f"\nفایل Test:")
print(f"  نام: {TEST_FILE.resolve()}")
print(f"  تعداد نمونه: {len(test_df)}")
print(f"  ابعاد: {test_df.shape}")

print("\nتوزیع کلاس‌ها در Test:")
print(test_df["Group"].value_counts())
