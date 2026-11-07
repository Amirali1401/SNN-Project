import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ۱. خواندن فایل پیش‌بینی‌ها که مدل شما ذخیره کرده است
predictions_path = "mdd_test_results/predictions.csv"
try:
    df = pd.read_csv(predictions_path)
except FileNotFoundError:
    print(f"خطا: فایل {predictions_path} پیدا نشد. مطمئن شوید مسیر درست است.")
    exit(1)

y_true = df["true_label"].to_numpy()
probabilities = df["probability_mdd"].to_numpy()

# ۲. پیدا کردن بهترین آستانه با استفاده از منحنی ROC و شاخص یودن (Youden’s J)
fpr, tpr, thresholds = roc_curve(y_true, probabilities)
# شاخص یودن: J = Sensitivity (TPR) + Specificity (1 - FPR) - 1
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"==========================================")
print(f"آستانه پیش‌فرض قبلی: 0.5")
print(f"بهترین آستانه بهینه محاسبه شده: {best_threshold:.4f}")
print(f"==========================================")

# ۳. ارزیابی مدل با آستانه جدید
y_pred_new = (probabilities >= best_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred_new, labels=[0, 1])
counts = np.bincount(y_true, minlength=2)

acc = accuracy_score(y_true, y_pred_new)
bal_acc = balanced_accuracy_score(y_true, y_pred_new)
prec = precision_score(y_true, y_pred_new, zero_division=0)
rec = recall_score(y_true, y_pred_new, zero_division=0)
f1 = f1_score(y_true, y_pred_new, zero_division=0)
spec = cm[0, 0] / counts[0] if counts[0] > 0 else 0.0
auc = roc_auc_score(y_true, probabilities)

print(f"\nنتایج مدل با آستانه بهینه ({best_threshold:.4f}):")
print(f"  Accuracy:          {acc:.4f}")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Precision (MDD):   {prec:.4f}")
print(f"  Recall (MDD):      {rec:.4f}")
print(f"  Specificity (Ctrl):{spec:.4f}")
print(f"  F1-Score:          {f1:.4f}")
print(f"  ROC-AUC:           {auc:.4f}")
print(f"\nماتریس درهم‌ریختگی جدید (سطرها واقعی، ستون‌ها پیش‌بینی):")
print(cm)

# ۴. ذخیره نتایج جدید در صورت نیاز
df["predicted_label_optimized"] = y_pred_new
df["predicted_class_optimized"] = np.where(y_pred_new == 0, "control", "mdd")
df.to_csv("mdd_test_results/predictions_optimized.csv", index=False)
print(f"\nفایل نتایج بهینه در 'mdd_test_results/predictions_optimized.csv' ذخیره شد.")
