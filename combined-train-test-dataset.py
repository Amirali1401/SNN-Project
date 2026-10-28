#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers

# -----------------------------
# ورودی‌ها و مسیرها
# -----------------------------
NPZ_PATH = "preprocessed_data.npz"
JSON_FEATURES_PATH = "fs_artifacts/selected_features.json"
OUT_DIR = "snn_results"

SEED = 42
BATCH_SIZE = 16          # به دلیل حجم نمونه مناسب‌تر است
EPOCHS = 200
LEARNING_RATE = 1e-3

# معماری Self-Normalizing
HIDDEN_UNITS = [64, 32]  # متناسب با ابعاد ۳۸ ورودی
ALPHA_DROPOUT = 0.10
L2 = 1e-4

os.makedirs(OUT_DIR, exist_ok=True)

# تثبیت رندوم سید
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# ۱. بارگذاری داده‌ها و نام ویژگی‌ها
# -----------------------------
print("[1] Loading preprocessed data and selected features...")
data = np.load(NPZ_PATH, allow_pickle=True)

# بررسی کلیدهای موجود در فایل NPZ
x_train_key = "X_train_scaled" if "X_train_scaled" in data.files else "X_train"
x_test_key = "X_test_scaled" if "X_test_scaled" in data.files else "X_test"
feature_names_key = "feature_names" if "feature_names" in data.files else None

X_train_all = data[x_train_key].astype(np.float32)
X_test_all = data[x_test_key].astype(np.float32)
y_train = data["y_train"].astype(np.int64)
y_test = data["y_test"].astype(np.int64)

# بارگذاری ۳۸ ویژگی منتخب از فایل JSON
with open(JSON_FEATURES_PATH, "r", encoding="utf-8") as f:
    fs_meta = json.load(f)
selected_probes = fs_meta["selected_features"]
label_mapping = fs_meta.get("label_mapping", {0: "bipolar", 1: "control", 2: "mdd"})

# اگر نام فیچرها در فایل npz ذخیره شده بود، بر اساس نام فیلتر می‌کنیم؛ در غیر این صورت ۳۸ تای اول
if feature_names_key:
    all_features = list(data[feature_names_key])
    selected_idx = [all_features.index(p) for p in selected_probes if p in all_features]
else:
    selected_idx = list(range(len(selected_probes)))

X_train = X_train_all[:, selected_idx]
X_test = X_test_all[:, selected_idx]

print(f"-> Selected {X_train.shape[1]} features out of {X_train_all.shape[1]}.")
print(f"-> Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -----------------------------
# ۲. محاسبه وزن کلاس‌ها (Imbalance)
# -----------------------------
classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
print(f"-> Class weights: {class_weight}")

# -----------------------------
# ۳. ساخت مدل SNN (SELU + AlphaDropout)
# -----------------------------
def build_snn(input_dim: int, num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,), name="Gene_Probes_Input")
    x = inp
    for i, units in enumerate(HIDDEN_UNITS, start=1):
        x = layers.Dense(
            units,
            activation="selu",
            kernel_initializer="lecun_normal",
            kernel_regularizer=regularizers.l2(L2),
            name=f"selu_dense_{i}"
        )(x)
        x = layers.AlphaDropout(ALPHA_DROPOUT, name=f"alpha_dropout_{i}")(x)

    out = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="lecun_normal",
        name="Output_Classification"
    )(x)

    model = models.Model(inp, out, name="Self_Normalizing_Net")
    return model

model = build_snn(input_dim=X_train.shape[1], num_classes=len(classes))
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# ۴. Callbackها و آموزش
# -----------------------------
ckpt_path = os.path.join(OUT_DIR, "best_snn_model.keras")

callbacks_list = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
        verbose=1
    )
]

print("\n[2] Training SNN model...")
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks_list,
    verbose=1
)

# -----------------------------
# ۵. ارزیابی نهایی روی داده‌های تست
# -----------------------------
print("\n[3] Evaluating on independent Test Set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_proba, axis=1)

target_names = [str(label_mapping.get(c, c)) for c in sorted(classes)]

print(f"\n==================== RESULTS ====================")
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("=================================================")

# ذخیره پیش‌بینی‌ها
np.savez(
    os.path.join(OUT_DIR, "test_predictions.npz"),
    y_test=y_test,
    y_pred=y_pred,
    y_proba=y_proba
)
print(f"\nArtifacts successfully saved to '{OUT_DIR}/'")
