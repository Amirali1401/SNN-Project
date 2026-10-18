from pathlib import Path
import csv
import gzip
import io
import re

import pandas as pd


# نام فایل ورودی را در صورت نیاز عوض کن
INPUT_FILE = Path("GSE98793_series_matrix (1).txt")


def read_lines(path):
    """خواندن فایل متنی معمولی یا فشردهٔ .gz"""
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.readlines()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


def get_group(sample_info):
    """
    تشخیص برچسب برای GSE98793:
    MDD در برابر Control.
    اگر برچسب در metadata با این عبارت‌ها فرق دارد،
    الگوهای این تابع را متناسب با فایل تغییر بده.
    """
    relevant_text = []

    for key, value in sample_info.items():
        key_lower = key.lower()

        # ابتدا مشخصات بالینی و نام منبع نمونه را بررسی می‌کنیم
        if "characteristics" in key_lower or "source_name" in key_lower:
            relevant_text.append(str(value))

    # اگر مشخصات بالینی/منبع موجود نبود، عنوان نمونه را هم بررسی کن
    if not relevant_text:
        relevant_text = [
            str(value)
            for key, value in sample_info.items()
            if "title" in key.lower()
        ]

    text = " ".join(relevant_text).lower()

    # کنترل را پیش از MDD بررسی می‌کنیم تا عباراتی مثل
    # "non-MDD control" به‌اشتباه بیمار برچسب نخورند.
    if re.search(r"\b(control|cntl|healthy)\b", text):
        return "Control"

    if re.search(r"\b(mdd|major depressive disorder)\b", text):
        return "MDD"

    return "Unknown"


# خواندن فایل و پیدا کردن بلوک جدول بیان
lines = read_lines(INPUT_FILE)

table_begin = next(
    (i for i, line in enumerate(lines)
     if line.strip().lower().startswith("!series_matrix_table_begin")),
    None
)
table_end = next(
    (i for i, line in enumerate(lines)
     if line.strip().lower().startswith("!series_matrix_table_end")),
    None
)

if table_begin is None or table_end is None:
    raise ValueError(
        "بلوک جدول بیان پیدا نشد. مطمئن شو فایل ورودی یک Series Matrix معتبر است."
    )

# خواندن metadata نمونه‌ها، از جمله شناسه و مشخصات بالینی
metadata = {}

for line in lines[:table_begin]:
    if line.startswith("!Sample_"):
        fields = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
        if fields:
            metadata[fields[0]] = fields[1:]

# جدول Series Matrix: ردیف‌ها پروب/ژن و ستون‌ها نمونه هستند
table_text = "".join(lines[table_begin + 1:table_end])

expression = pd.read_csv(
    io.StringIO(table_text),
    sep="\t",
    index_col=0,
    low_memory=False
)

# تبدیل مقادیر بیان به عدد؛ مقادیر غیرعددی به مقدار خالی تبدیل می‌شوند
expression = expression.apply(pd.to_numeric, errors="coerce")

# ساخت metadata برای هر شناسهٔ نمونه
sample_ids = [
    str(x).strip()
    for x in metadata.get("!Sample_geo_accession", [])
]

metadata_by_sample = {}

for i, sample_id in enumerate(sample_ids):
    metadata_by_sample[sample_id] = {
        key: values[i] if i < len(values) else ""
        for key, values in metadata.items()
    }

# جابه‌جایی ماتریس: هر ردیف نمونه و هر ستون یک ویژگی ژنی/پروبی
output = expression.T
output.insert(0, "ID_REF", output.index.astype(str))

# اضافه‌کردن برچسب هر نمونه
output["Group"] = [
    get_group(metadata_by_sample.get(str(sample_id), {}))
    for sample_id in expression.columns
]

# ساخت نام خروجی از روی نام فایل ورودی:
# GSE98793_series_matrix.txt.gz -> GSE98793_series_matrix.csv
output_path = INPUT_FILE
if output_path.suffix.lower() == ".gz":
    output_path = output_path.with_suffix("")

output_path = output_path.with_suffix(".csv")

# ذخیرهٔ CSV
output.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"CSV ساخته شد: {output_path.resolve()}")
print("\nتعداد نمونه‌ها در هر گروه:")
print(output["Group"].value_counts(dropna=False))

unknown = output.loc[output["Group"] == "Unknown", "ID_REF"].tolist()
if unknown:
    print(
        "\nهشدار: برای این نمونه‌ها برچسب خودکار پیدا نشد؛ "
        "metadata آن‌ها را بررسی کن:"
    )
    print(unknown)
