import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, AlphaDropout
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report




import os
import json
import joblib


# تنظیم سید برای تکرارپذیری دقیق
np.random.seed(42)
tf.random.set_seed(42)

# ۱. لیست ۳۸ ژن انتخاب شده از الگوریتم MOPSO
selected_genes = [
    '204493_at',
    '205404_at',
    '221780_s_at',
]

# ۲. بارگذاری دیتاست
file_path = "Top - 100 - Bipolar.csv"
df = pd.read_csv(file_path)

if 'ID_REF' in df.columns:
    df = df.drop(columns=['ID_REF'])

target_col = df.columns[-1]
print("تعداد ستون‌های df:", len(df.columns))
print("نمونه ستون‌های df:", df.columns.tolist()[:20])
print("نمونه selected_genes:", selected_genes[:10])

missing = [g for g in selected_genes if g not in df.columns]
print("چند تا از ژن‌ها پیدا نشدند:", len(missing), "از", len(selected_genes))
print("نمونه‌های گمشده:", missing[:10])


# فیلتر داده‌ها بر اساس ۳۸ ژن انتخاب شده
X = df[selected_genes].values
y = LabelEncoder().fit_transform(df[target_col].values)

# استانداردسازی داده‌ها (پیش‌نیاز اساسی SNN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ۳. تابع ساخت مدل SNN استاندارد (SELU + LeCun Normal + AlphaDropout)
def build_snn_model(input_dim):
    model = Sequential([
        Dense(64, activation='selu', kernel_initializer='lecun_normal', input_shape=(input_dim,)),
        AlphaDropout(0.1),
        Dense(32, activation='selu', kernel_initializer='lecun_normal'),
        AlphaDropout(0.1),
        Dense(16, activation='selu', kernel_initializer='lecun_normal'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ۴. ارزیابی با Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []

print("="*60)
print(f"Training SNN Model with {len(selected_genes)} Selected Genes...")
print("="*60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # ساخت و آموزش مدل
    model = build_snn_model(input_dim=X_train.shape[1])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=8,
        verbose=0
    )
    
    # پیش‌بینی روی داده اعتبارسنجی
    y_prob = model.predict(X_val, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    
    # محاسبه متریک‌ها برای هر فولد
    acc_list.append(accuracy_score(y_val, y_pred))
    prec_list.append(precision_score(y_val, y_pred, zero_division=0))
    rec_list.append(recall_score(y_val, y_pred, zero_division=0))
    f1_list.append(f1_score(y_val, y_pred, zero_division=0))
    auc_list.append(roc_auc_score(y_val, y_prob))
    
    print(f"Fold {fold} -> Acc: {acc_list[-1]*100:.2f}% | F1: {f1_list[-1]:.4f} | AUC: {auc_list[-1]:.4f}")

# ۵. گزارش میانگین نتایج نهایی
print("="*60)
print("FINAL SNN CROSS-VALIDATION RESULTS (38 Genes):")
print("="*60)
print(f"Accuracy : {np.mean(acc_list)*100:.2f}% (± {np.std(acc_list)*100:.2f}%)")
print(f"Precision: {np.mean(prec_list):.4f}")
print(f"Recall   : {np.mean(rec_list):.4f}")
print(f"F1-Score : {np.mean(f1_list):.4f}")
print(f"ROC-AUC  : {np.mean(auc_list):.4f}")
print("="*60)



os.makedirs("bipolar_artifacts", exist_ok=True)

# اگر نام مدل شما model است، از همین متغیر استفاده کنید.
# در صورت متفاوت بودن نام متغیر، model را با نام واقعی مدل عوض کنید.
model.save(
    "bipolar_artifacts/bipolar_snn_model.keras"
)

joblib.dump(
    scaler,
    "bipolar_artifacts/bipolar_scaler.pkl"
)

with open(
    "bipolar_artifacts/bipolar_selected_genes.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        selected_genes,
        f,
        ensure_ascii=False,
        indent=4
    )

print("Bipolar artifacts saved successfully.")
