from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
OUT = Path("combined_shap_results")
tf.keras.utils.set_random_seed(SEED)


def locate(*paths):
    for path in paths:
        if Path(path).is_file():
            return Path(path)
    raise FileNotFoundError(
        "فایل پیدا نشد؛ مسیر را اصلاح کنید: " + " | ".join(paths)
    )


CSV_PATH = locate("merged_MDD_Bipolar_100probes_labeled.csv")
NPZ_PATH = locate("preprocessed_data.npz")
JSON_PATH = locate(
    "selected_features.json",
    "fs_artifacts/selected_features.json",
)
MODEL_PATH = locate(
    "best_snn_model.keras",
    "snn_results/best_snn_model.keras",
)
REFERENCE_PATH = locate(
    "test_predictions.npz",
    "snn_results/test_predictions.npz",
)

# خواندن مشخصات و آرایه‌های ذخیره‌شده
with JSON_PATH.open(encoding="utf-8") as f:
    metadata = json.load(f)

selected = metadata["selected_features"]
label_mapping = metadata["label_mapping"]

with np.load(NPZ_PATH, allow_pickle=False) as data:
    Xtr_all = data["X_train"].astype(np.float32)
    Xte_all = data["X_test"].astype(np.float32)
    ytr = data["y_train"].astype(np.int64)
    yte = data["y_test"].astype(np.int64)

df = pd.read_csv(CSV_PATH)
protected = {"Sample_ID", "Group", "Source"}
probes = [c for c in df.columns if c not in protected]

if (
    len(probes) != 100
    or len(selected) != 38
    or len(set(selected)) != 38
    or set(selected) - set(probes)
):
    raise ValueError("تعداد یا نام پروب‌ها با فایل‌های بررسی‌شده تطابق ندارد.")

raw = df[probes].to_numpy(dtype=np.float64)
encoded = df["Group"].map(label_mapping)

if encoded.isna().any() or not np.isfinite(raw).all():
    raise ValueError("برچسب ناشناخته یا مقدار غیرعددی/نامعتبر در داده وجود دارد.")

y_csv = encoded.to_numpy(dtype=np.int64)

# این بازسازی فقط برای تأیید نام و ترتیب ستون‌هاست.
# ورودی مدل همچنان از آرایه‌های اصلی گرفته می‌شود؛ دوباره مقیاس‌بندی نمی‌شود.
train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    stratify=y_csv,
    random_state=SEED,
)

scaler = StandardScaler().fit(raw[train_idx])
rebuilt_train = scaler.transform(raw[train_idx])
rebuilt_test = scaler.transform(raw[test_idx])


def matches(a, b):
    return (
        a.shape == b.shape
        and np.allclose(a, b, rtol=1e-5, atol=1e-5)
    )


if not (
    matches(rebuilt_train, Xtr_all)
    and matches(rebuilt_test, Xte_all)
    and np.array_equal(y_csv[train_idx], ytr)
    and np.array_equal(y_csv[test_idx], yte)
):
    raise RuntimeError(
        "ترتیب نمونه‌ها، ستون‌ها یا پیش‌پردازش بازتولید نشد. "
        "تحلیل متوقف شد تا پروب‌ها اشتباه نام‌گذاری نشوند."
    )

# مطابق شاخه‌ای از کد آموزش که بدون نام ویژگی‌ها اجرا می‌شود
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

if int(model.input_shape[-1]) != 38 or int(model.output_shape[-1]) != 3:
    raise ValueError("مدل باید ۳۸ ورودی و ۳ خروجی داشته باشد.")

Xtr = Xtr_all[:, :38]
Xte = Xte_all[:, :38]
feature_names = probes[:38]


def predict(x):
    x = np.asarray(x, dtype=np.float32)
    return model(x, training=False).numpy()


probabilities = predict(Xte)

# تأیید اینکه مدل و ترتیب ورودی، پیش‌بینی مرجع را بازتولید می‌کنند
with np.load(REFERENCE_PATH, allow_pickle=False) as reference:
    reference_y = reference["y_test"]
    reference_probabilities = reference["y_proba"]

if not (
    np.array_equal(yte, reference_y)
    and matches(probabilities, reference_probabilities)
):
    raise RuntimeError(
        "پیش‌بینی مرجع بازتولید نشد. "
        "مدل یا ترتیب ورودی با فایل پیش‌بینی یکسان نیست؛ تحلیل متوقف شد."
    )

OUT.mkdir(exist_ok=True)

pd.DataFrame({
    "input_position": np.arange(1, 39),
    "probe": feature_names,
    "listed_in_selected_features": [
        name in selected for name in feature_names
    ],
}).to_csv(OUT / "verified_model_inputs.csv", index=False)

# پس‌زمینه فقط از داده‌های آموزش انتخاب می‌شود.
_, background_idx = train_test_split(
    np.arange(len(Xtr)),
    test_size=40,
    stratify=ytr,
    random_state=SEED,
)
background = Xtr[background_idx]

np.random.seed(SEED)
explainer = shap.KernelExplainer(
    predict,
    background,
    link="identity",
)

# برآورد تقریبی شاپ برای تمام ۵۲ نمونه آزمون
result = explainer.shap_values(
    Xte,
    nsamples=512,
    l1_reg=0.0,
)

values = (
    np.stack(result, axis=-1)
    if isinstance(result, list)
    else np.asarray(result)
)

if values.shape != (len(Xte), 38, 3):
    raise RuntimeError(
        f"ابعاد خروجی شاپ غیرمنتظره است: {values.shape}"
    )

base = np.asarray(explainer.expected_value).reshape(3)
residual = np.max(
    np.abs(base + values.sum(axis=1) - probabilities)
)

if not np.isfinite(values).all() or residual > 1e-4:
    raise RuntimeError("کنترل جمع‌پذیری مقادیر شاپ تأیید نشد.")

class_names = ["bipolar", "control", "mdd"]
mean_abs = np.abs(values).mean(axis=0)

importance = pd.DataFrame({
    "probe": feature_names,
    "mean_abs_all": mean_abs.mean(axis=1),
    **{
        f"mean_abs_{name}": mean_abs[:, k]
        for k, name in enumerate(class_names)
    },
}).sort_values("mean_abs_all", ascending=False)

importance.to_csv(
    OUT / "shap_feature_importance.csv",
    index=False,
)

np.savez_compressed(
    OUT / "shap_values.npz",
    shap_values=values,
    expected_values=base,
    X_test=Xte,
    y_test=yte,
    probabilities=probabilities,
    feature_names=np.asarray(feature_names, dtype=str),
    class_names=np.asarray(class_names, dtype=str),
    test_sample_ids=df.iloc[test_idx]["Sample_ID"].to_numpy(dtype=str),
    background_train_indices=background_idx,
)

for k, name in enumerate(class_names):
    plt.figure()
    shap.summary_plot(
        values[:, :, k],
        Xte,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.savefig(
        OUT / f"shap_summary_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

top = importance.head(20).iloc[::-1]
plt.figure(figsize=(9, 7))
plt.barh(top["probe"], top["mean_abs_all"])
plt.xlabel("Mean absolute SHAP value")
plt.tight_layout()
plt.savefig(OUT / "shap_global_importance.png", dpi=300)
plt.close()

print("تحلیل تمام شد. مسیر خروجی:", OUT.resolve())
print("بیشترین خطای جمع‌پذیری:", residual)
print(importance.head(10).to_string(index=False))
