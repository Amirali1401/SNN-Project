from pathlib import Path
from collections import Counter
import csv
import gzip
import io

import pandas as pd


INPUT_FILE = Path("GSE98793_series_matrix (1).txt")
OUTPUT_FILE = Path("GSE98793_labeled.csv")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"فایل ورودی پیدا نشد: {INPUT_FILE.resolve()}"
        )

    open_file = gzip.open if INPUT_FILE.suffix == ".gz" else open

    sample_ids = None
    subject_groups = None
    table_lines = []
    inside_table = False
    table_finished = False

    with open_file(INPUT_FILE, "rt", encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                inside_table = True
                continue

            if line.startswith("!series_matrix_table_end"):
                table_finished = True
                break

            if inside_table:
                table_lines.append(line)
                continue

            row = next(csv.reader([line], delimiter="\t"))
            if not row:
                continue

            key = row[0]
            values = [value.strip() for value in row[1:]]

            if key == "!Sample_geo_accession":
                sample_ids = values

            elif key == "!Sample_characteristics_ch1":
                is_subject_group = any(
                    value.lower().startswith("subject group:")
                    for value in values
                )

                if is_subject_group:
                    if subject_groups is not None:
                        raise ValueError(
                            "بیش از یک ردیف subject group پیدا شد."
                        )
                    subject_groups = values

    if sample_ids is None or subject_groups is None:
        raise ValueError(
            "شناسهٔ نمونه‌ها یا فیلد subject group پیدا نشد."
        )

    if len(sample_ids) != len(subject_groups):
        raise ValueError(
            "تعداد شناسه‌های نمونه با تعداد برچسب‌ها برابر نیست."
        )

    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("شناسهٔ نمونهٔ تکراری در متادیتا وجود دارد.")

    if not table_lines or not table_finished:
        raise ValueError("جدول بیان ژن پیدا نشد یا ناقص است.")

    # استخراج برچسب فقط از وضعیت بالینی ثبت‌شده
    labels = []

    for sample_id, value in zip(sample_ids, subject_groups):
        prefix, separator, description = value.partition(":")

        if (
            not separator
            or prefix.strip().lower() != "subject group"
        ):
            raise ValueError(
                f"متادیتای گروه نامعتبر برای {sample_id}: {value}"
            )

        group_code = description.split(";", 1)[0].strip().upper()

        if group_code == "CNTL":
            labels.append("control")
        elif group_code == "CASE":
            labels.append("mdd")
        else:
            raise ValueError(
                f"گروه ناشناخته برای {sample_id}: {value}"
            )

    label_by_id = dict(zip(sample_ids, labels))

    # جدول اصلی: پروب‌ها در سطر و نمونه‌ها در ستون
    expression = pd.read_csv(
        io.StringIO("".join(table_lines)),
        sep="\t",
        index_col=0,
    )

    expression.columns = expression.columns.astype(str).str.strip()

    if expression.columns.duplicated().any():
        raise ValueError("شناسهٔ نمونهٔ تکراری در جدول بیان وجود دارد.")

    if expression.index.has_duplicates:
        raise ValueError("شناسهٔ پروب تکراری در جدول بیان وجود دارد.")

    if set(expression.columns) != set(sample_ids):
        raise ValueError(
            "شناسه‌های نمونه در جدول بیان و متادیتا یکسان نیستند."
        )

    # خروجی: هر سطر یک نمونه، هر ستون یک پروب
    output = expression.T
    output.index.name = "ID_REF"

    # اتصال برچسب بر اساس شناسه، نه موقعیت ردیف
    output["Group"] = [
        label_by_id[sample_id] for sample_id in output.index
    ]

    output = output.reset_index()

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nفایل ساخته شد: {OUTPUT_FILE.resolve()}")
    print(f"ابعاد داده: {output.shape}")
    print(f"تعداد ویژگی‌ها: {expression.shape[0]}")

    print("\nتوزیع کلاس‌ها:")
    print(output["Group"].value_counts())

    expected = {"control": 64, "mdd": 128}
    if dict(Counter(labels)) != expected:
        print(
            "\nتوجه: تعداد کلاس‌ها با خروجی قبلی "
            "شما (۶۴ کنترل و ۱۲۸ بیمار) متفاوت است."
        )


if __name__ == "__main__":
    main()
