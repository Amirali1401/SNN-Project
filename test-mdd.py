import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="ارزیابی مدل MDD روی داده تست")
    parser.add_argument(
        "--test-csv",
        type=str,
        default="mdd_test.csv",
        help="مسیر فایل تست CSV",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="mdd_model_artifacts",
        help="پوشه حاوی مدل و ابزارهای پیش‌پردازش",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mdd_test_results",
        help="مسیر ذخیره نتایج تست",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7216,
        help="آستانه تصمیم‌گیری برای کلاس MDD (پیش‌فرض: 0.7216)",
    )
    return parser.parse_args()


def load_artifacts(artifacts_dir: Path):
    print(f"بارگذاری تنظیمات و ابزارهای پیش‌پردازش از: {artifacts_dir}")

    # ۱. مدل
    model_path = artifacts_dir / "snn_model.keras"
    if not model_path.exists():
        model_path = artifacts_dir / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"فایل مدل در {artifacts_dir} یافت نشد.")
    model = tf.keras.models.load_model(model_path)

    # ۲. اسکیلر و Imputer
    scaler_path = artifacts_dir / "scaler.pkl"
    imputer_path = artifacts_dir / "imputer.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    imputer = joblib.load(imputer_path) if imputer_path.exists() else None

    # ۳. لیست ژن‌ها
    genes_path = artifacts_dir / "selected_genes.json"
    meta_path = artifacts_dir / "feature_metadata.json"
    if genes_path.exists():
        with open(genes_path, "r", encoding="utf-8") as f:
            feature_genes = json.load(f)
    elif meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            feature_genes = meta.get("feature_names", meta.get("selected_genes", []))
    else:
        raise FileNotFoundError("لیست ژن‌های انتخابی یافت نشد.")

    # ۴. مپینگ برچسب‌ها
    mapping_path = artifacts_dir / "label_mapping.json"
    encoder_path = artifacts_dir / "label_encoder.pkl"
    label_map = None
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    elif encoder_path.exists():
        encoder = joblib.load(encoder_path)
        label_map = {cls: idx for idx, cls in enumerate(encoder.classes_)}

    return model, scaler, imputer, feature_genes, label_map


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # بارگذاری فایل تست
    test_csv = Path(args.test_csv)
    if not test_csv.exists():
        raise FileNotFoundError(f"فایل داده تست یافت نشد: {test_csv}")
    df_test = pd.read_csv(test_csv)

    model, scaler, imputer, feature_genes, label_map = load_artifacts(artifacts_dir)

    # جداسازی لیبل
        # جداسازی لیبل
    target_col = None
    for col in ["target", "label", "condition", "group", "Group", "status", "diagnosis", "class"]:
        if col in df_test.columns:
            target_col = col
            break


    if target_col is None:
        raise ValueError("ستون برچسب (مانند target یا label) در فایل تست پیدا نشد.")

    y_raw = df_test[target_col]
    
    # نگاشت برچسب به 0 و 1
    if label_map is not None:
        y_true = y_raw.map(label_map).values
    else:
        # نگاشت پیش‌فرض: control -> 0 و بقیه -> 1
        y_true = (
            y_raw.astype(str)
            .str.lower()
            .apply(lambda x: 0 if "control" in x or x == "0" else 1)
            .values
        )

    # انتخاب ویژگی‌ها
    available_genes = [g for g in feature_genes if g in df_test.columns]
    X_test = df_test[available_genes].copy()

    # پر کردن ستون‌های مفقود اگر وجود داشتند
    for missing_gene in set(feature_genes) - set(available_genes):
        X_test[missing_gene] = 0.0
    X_test = X_test[feature_genes]

    # اعمال Imputer و Scaler
    if imputer is not None:
        X_test = imputer.transform(X_test)
    if scaler is not None:
        X_test = scaler.transform(X_test)

    # پیش‌بینی احتمال
    probs = model.predict(X_test)
    if probs.ndim > 1 and probs.shape[1] > 1:
        probs = probs[:, 1]
    else:
        probs = probs.ravel()

    # اعمال آستانه تصمیم‌گیری
    preds = (probs >= args.threshold).astype(int)

    # محاسبه معیارها
    acc = accuracy_score(y_true, preds)
    bacc = balanced_accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 45)
    print(f"نتایج ارزیابی تست با آستانه {args.threshold}:")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bacc:.4f}")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"  Specificity:       {spec:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  ROC-AUC:           {auc:.4f}")
    print("\nماتریس درهم‌ریختگی (Confusion Matrix):")
    print(cm)
    print("=" * 45)

    # ذخیره پیش‌بینی‌ها
    results_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_prob": probs,
            "predicted_label": preds,
        }
    )
    save_path = output_dir / "predictions.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nفایل نتایج در '{save_path}' ذخیره شد.")


if __name__ == "__main__":
    main()
