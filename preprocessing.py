import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

PROTECTED_COLS = ["Sample_ID", "Group", "Source"]


def load_merged_dataset(path):
    """بارگذاری دیتاست ادغام‌شده و برقراری پره‌نیازهای اولیه."""
    df = pd.read_csv(path)
    assert "Group" in df.columns, "Group column not found!"
    assert "Sample_ID" in df.columns, "Sample_ID column not found!"
    return df


def get_feature_columns(df):
    """ستون‌های پروب (خارج از ستون‌های غیر-feature) را برمی‌گرداند."""
    feature_cols = [c for c in df.columns if c not in PROTECTED_COLS]
    assert len(feature_cols) == 100, f"Expected 100 probe features, got {len(feature_cols)}"
    return feature_cols


def fit_preprocess_on_train(train_df, feature_cols):
    """
    فقط روی Train fit می‌کند (بدون نشت داده):
    - LabelEncoder روی Group
    - StandardScaler روی X
    """
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Group"].astype(str))

    X_train = train_df[feature_cols].astype(float).values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    return X_train_scaled, y_train, scaler, le


def transform_with_fitted(test_df, feature_cols, scaler, label_encoder):
    """فقط transform (بدون fit) برای Test."""
    y_test = label_encoder.transform(test_df["Group"].astype(str))
    X_test = test_df[feature_cols].astype(float).values
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, y_test


def label_mapping(label_encoder):
    """نگاشت کلاس به عدد را برمی‌گرداند."""
    return {cls: int(i) for i, cls in enumerate(label_encoder.classes_)}


if __name__ == "__main__":
    # ---- تنظیم مسیر فایل تو ----
    data_path = "/home/amirali/Desktop/SNN-Project/merged_MDD_Bipolar_100probes_labeled.csv"
    df = load_merged_dataset(data_path)
    feature_cols = get_feature_columns(df)

    # ---- مثال: Split تصادفی برای تست کد ----
    # (اگر split آماده‌ی خودت را داری، این دو خط را با خواندن فایل train/test جایگزین کن)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Group"], random_state=42)

    X_train_scaled, y_train, scaler, le = fit_preprocess_on_train(train_df, feature_cols)
    X_test_scaled, y_test = transform_with_fitted(test_df, feature_cols, scaler, le)

    print("X_train_scaled:", X_train_scaled.shape)
    print("X_test_scaled:", X_test_scaled.shape)
    print("Classes:", le.classes_)
    print("Label map:", label_mapping(le))

    np.savez(
        "preprocessed_data.npz",
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test
        )
    print("Saved: preprocessed_data.npz")

