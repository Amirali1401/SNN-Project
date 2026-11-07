import pandas as pd
import numpy as np
import json
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ۱. مسیر فایل‌ها (اگر مسیر دیگری داری اینجا تغییر بده)
TRAIN_FILE = "mdd_train.csv"
GENES_FILE = "mdd_artifacts/selected_genes.json"
OUTPUT_SYNTHETIC_FILE = "synthetic_samples_generated_by_smote.csv"

# ۲. خواندن ژن‌های منتخب
with open(GENES_FILE, 'r', encoding='utf-8') as f:
    obj = json.load(f)
selected_genes = obj if isinstance(obj, list) else obj.get('selected_genes')

# ۳. خواندن داده‌های آموزشی
df = pd.read_csv(TRAIN_FILE)

# پیدا کردن ستون تارگت/لیبل
target_col = [c for c in df.columns if str(c).lower() in ["group", "label", "target", "class", "disease"]][0]

# فیلتر کردن ژن‌ها و تبدیل لیبل به 0 و 1
X_raw = df[selected_genes]
y = df[target_col].astype(str).str.lower().map(
    lambda x: 0 if x in ["control", "hc", "cntl", "ctrl", "0", "healthy"] else 1
).to_numpy()

# ۴. پیش‌پردازش استاندارد (Impute + Scale) دقیقا مثل اسکریپت آموزش
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(imputer.fit_transform(X_raw))

# ۵. اجرای SMOTE
original_sample_count = len(df)
smote = SMOTE(k_neighbors=3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# ۶. برگرداندن داده‌ها از حالت مقیاس‌شده به مقادیر واقعی بیان ژن (Inverse Transform)
X_resampled_original_scale = scaler.inverse_transform(X_resampled)

# ۷. استخراج *فقط* نمونه‌های تولید شده توسط SMOTE
# SMOTE نمونه‌های جدید را دقیقا به انتهای آرایه اضافه می‌کند
synthetic_X = X_resampled_original_scale[original_sample_count:]
synthetic_y = y_resampled[original_sample_count:]

# ذخیره در DataFrame
df_synthetic = pd.DataFrame(synthetic_X, columns=selected_genes)
df_synthetic["Label"] = ["MDD" if label == 1 else "Control" for label in synthetic_y]

# ذخیره در فایل CSV
df_synthetic.to_csv(OUTPUT_SYNTHETIC_FILE, index=False)

print(f"✅ تعداد نمونه‌های واقعی اولیه: {original_sample_count}")
print(f"✨ تعداد کل نمونه‌های تولید شده با SMOTE: {len(df_synthetic)}")
print(f"📊 تفکیک نمونه‌های تولید شده: {pd.Series(synthetic_y).value_counts().to_dict()}")
print(f"💾 نمونه‌های تولید شده با مقادیر واقعی ژن‌ها در فایل زیر ذخیره شدند:\n   -> {OUTPUT_SYNTHETIC_FILE}")

# نمایش ۵ سطر اول نمونه‌های تولید شده در کنسول
print("\nپیش‌نمایش ۵ نمونه اول تولید شده:")
print(df_synthetic.head())
