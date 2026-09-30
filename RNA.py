from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# تنظیم مسیر فایل‌ها
# ---------------------------------------------------------------------
SELECTED_GENES_FILE = Path("selected_genes.csv")
COUNTS_FILE = Path("GSE290452_raw_counts.tsv")

# این فایل، فهرست ژن‌ها با ستون gene_name و در صورت وجود rank است.
# فایل خام شمارش باید ستون Gene و ستون‌هایی برای نمونه‌ها داشته باشد.


# ---------------------------------------------------------------------
# خواندن فهرست ژن‌های منتخب
# ---------------------------------------------------------------------
selected_df = pd.read_csv(SELECTED_GENES_FILE)

if "gene_name" not in selected_df.columns:
    raise ValueError(
        f"ستون 'gene_name' در {SELECTED_GENES_FILE} پیدا نشد. "
        f"ستون‌های موجود: {selected_df.columns.tolist()}"
    )

selected_genes = (
    selected_df["gene_name"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

if not selected_genes:
    raise ValueError("فهرست ژن‌های منتخب خالی است.")

print("Selected genes:", selected_genes)
print("Number of selected genes:", len(selected_genes))


# ---------------------------------------------------------------------
# خواندن ماتریس خام شمارش
# ---------------------------------------------------------------------
raw_counts = pd.read_csv(
    COUNTS_FILE,
    sep="\t",
    index_col="Gene"
)

# پاک‌سازی شناسه ژن‌ها برای تطبیق با فهرست منتخب
raw_counts.index = raw_counts.index.astype(str).str.strip()

# تبدیل داده‌ها به عدد؛ مقادیر نامعتبر به NaN تبدیل می‌شوند
raw_counts = raw_counts.apply(pd.to_numeric, errors="coerce")

if raw_counts.isna().any().any():
    raise ValueError(
        "در ماتریس شمارش مقادیر غیرعددی یا خالی وجود دارد. "
        "ساختار فایل خام را بررسی کن."
    )

# اگر نام ژنی چند بار تکرار شده، شمارش‌ها را با هم جمع می‌کنیم
raw_counts = raw_counts.groupby(level=0).sum()

# پیدا کردن ژن‌های موجود و ناموجود
available_genes = [
    gene for gene in selected_genes
    if gene in raw_counts.index
]
missing_genes = [
    gene for gene in selected_genes
    if gene not in raw_counts.index
]

print("Available selected genes:", len(available_genes))
if missing_genes:
    print("Genes not found in raw counts:", missing_genes)

if not available_genes:
    raise ValueError(
        "هیچ‌کدام از ژن‌های منتخب در فایل خام شمارش پیدا نشدند. "
        "نام ژن‌ها یا فایل ورودی را بررسی کن."
    )

# حفظ ترتیب ژن‌ها مطابق selected_genes.csv
counts = raw_counts.loc[available_genes].copy()


# ---------------------------------------------------------------------
# فیلتر ژن‌های کم‌بیان
# ژن باید در حداقل 3 نمونه، شمارش حداقل 10 داشته باشد.
# ---------------------------------------------------------------------
keep = (counts >= 10).sum(axis=1) >= 3
counts_filtered = counts.loc[keep].copy()

if counts_filtered.empty:
    raise ValueError(
        "پس از >=یلتر بیان، هیچ ژنی باقی نماند. "
        "آستانه count >= 10 یا تعداد نمونه‌ها را بررسی کن."
    )

print("Genes after expression filter:", counts_filtered.shape[0])


# ---------------------------------------------------------------------
# تبدیل به log2-CPM
# ---------------------------------------------------------------------
library_size = counts_filtered.sum(axis=0)

if (library_size <= 0).any():
    bad_samples = library_size[library_size <= 0].index.tolist()
    raise ValueError(f"Library size صفر یا منفی برای این نمونه‌ها: {bad_samples}")

cpm = counts_filtered.div(library_size, axis=1) * 1_000_000
log_counts = np.log2(cpm + 1)

# ساختار X: سطرها نمونه‌ها، ستون‌ها ژن‌ها
X = log_counts.T
X.index = X.index.astype(str)

print("Samples:", X.shape[0])
print("Features before CV selection:", X.shape[1])
print(X.head())


# ---------------------------------------------------------------------
# برچسب‌ها
# ترتیب برچسب‌ها باید دقیقاً با ترتیب ستون‌های فایل خام شمارش یکی باشد.
# ---------------------------------------------------------------------
y = np.array([
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    0, 0, 0
])

y_lithium = np.array([
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1
])

groups = np.array([
    "clone16", "clone20", "clone21",
    "WT_rep1", "WT_rep2", "WT_rep3",
    "clone16", "clone20", "clone21",
    "WT_rep1", "WT_rep2", "WT_rep3"
])

if len(y) != len(X):
    raise ValueError(
        f"تعداد برچسب‌ها ({len(y)}) با تعداد نمونه‌های X ({len(X)}) برابر نیست."
    )

if len(y_lithium) != len(X):
    raise ValueError(
        f"تعداد برچسب‌های lithium ({len(y_lithium)}) "
        f"با تعداد نمونه‌های X ({len(X)}) برابر نیست."
    )

if len(groups) != len(X):
    raise ValueError(
        f"تعداد groupها ({len(groups)}) با تعداد نمونه‌های X ({len(X)}) برابر نیست."
    )

# هشدار مهم: این چاپ ترتیب واقعی نمونه‌ها را نشان می‌دهد.
# y و groups باید دقیقاً با این ترتیب هماهنگ شده باشند.
print("Sample order in X:")
print(X.index.tolist())


# ---------------------------------------------------------------------
# Cross-validation و انتخاب ویژگی
# انتخاب ویژگی و StandardScaler داخل هر fold fit می‌شوند تا leakage رخ ندهد.
# ---------------------------------------------------------------------
class_counts = np.bincount(y)
smallest_class_count = class_counts[class_counts > 0].min()

if smallest_class_count < 3:
    raise ValueError(
        "برای StratifiedKFold با n_splits=3، هر کلاس باید حداقل 3 نمونه داشته باشد."
    )

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # حداکثر 100 ژن، یا همه ژن‌ها اگر تعدادشان کمتر از 100 باشد
    k = min(100, X_train.shape[1])

    feature_selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_test_selected = feature_selector.transform(X_test)

    selected_mask = feature_selector.get_support()
    fold_selected_genes = X.columns[selected_mask].tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    fold_results.append({
        "fold": fold,
        "train_indices": train_idx,
        "test_indices": test_idx,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "selected_genes": fold_selected_genes,
        "scaler": scaler,
        "feature_selector": feature_selector,
    })

    print(f"\nFold {fold}")
    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
    print("Selected genes:", fold_selected_genes)
