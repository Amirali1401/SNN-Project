from pathlib import Path
from collections import Counter
import csv
import gzip

INPUT_FILE = Path("GSE98793_series_matrix (1).txt")

open_file = gzip.open if INPUT_FILE.suffix == ".gz" else open
found = False

with open_file(INPUT_FILE, "rt", encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter="\t")

    for row in reader:
        if not row:
            continue

        if row[0].startswith("!series_matrix_table_begin"):
            break

        if row[0] != "!Sample_characteristics_ch1":
            continue

        groups = [
            value.strip()
            for value in row[1:]
            if value.strip().lower().startswith("subject group:")
        ]

        if groups:
            found = True
            print("\nگروه‌های ثبت‌شده در متادیتا:")
            for group, count in Counter(groups).items():
                print(f"{group}  -->  {count} samples")

            print(f"\nتعداد کل برچسب‌های این ردیف: {len(groups)}")

if not found:
    raise ValueError("فیلد subject group در متادیتا پیدا نشد.")
