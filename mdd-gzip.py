
# import gzip
# import pandas as pd

# matrix_file = "GSE98793_series_matrix.txt.gz"
# probes_file = "Top - 100 - Bipolar.csv"
# output_file = "GSE98793_MDD_100_probes.csv"

# # استخراج شناسه‌های 100 پروب از ستون‌های فایل Bipolar
# bipolar = pd.read_csv(probes_file)

# probe_columns = [
#     col for col in bipolar.columns
#     if col not in {"_REF", "ID_REF", "Group"}
# ]

# probe_set = set(probe_columns)

# print(f"تعداد پروب‌های موردنیاز: {len(probe_set)}")

# # پیدا کردن بخش اصلی ماتریس در فایل Series Matrix
# rows = []
# header = None
# table_started = False

# with gzip.open(matrix_file, "rt", encoding="utf-8", errors="replace") as f:
#     for line in f:
#         line = line.rstrip("\n")

#         if line.startswith("!series_matrix_table_begin"):
#             table_started = True
#             continue

#         if line.startswith("!series_matrix_table_end"):
#             break

#         if not table_started:
#             continue

#         fields = line.split("\t")

#         if header is None:
#             header = [x.strip('"') for x in fields]
#             continue

#         probe_id = fields[0].strip('"')

#         if probe_id in probe_set:
#             rows.append(fields)

# if not rows:
#     raise RuntimeError(
#         "هیچ پروبی پیدا نشد. احتمالاً شناسه‌ها یا فرمت فایل متفاوت است."
#     )

# # ساخت ماتریس: ردیف‌ها پروب و ستون‌ها نمونه هستند
# matrix = pd.DataFrame(rows, columns=header)

# matrix.iloc[:, 0] = matrix.iloc[:, 0].astype(str).str.strip('"')
# matrix = matrix.set_index(matrix.columns[0])

# # تبدیل به ساختار مشابه فایل Bipolar:
# # هر ردیف = یک نمونه، هر ستون = یک پروب
# output = matrix.T.reset_index()
# output = output.rename(columns={"index": "Sample_ID"})

# # مرتب‌سازی ستون‌ها بر اساس ترتیب فایل Bipolar
# available_probes = [p for p in probe_columns if p in output.columns]
# output = output[["Sample_ID"] + available_probes]

# # تبدیل مقادیر به عدد
# for col in available_probes:
#     output[col] = pd.to_numeric(output[col], errors="coerce")

# output.to_csv(output_file, index=False)

# print(f"پروب‌های پیدا شده: {len(available_probes)} از {len(probe_columns)}")
# print(f"تعداد نمونه‌های MDD: {len(output)}")
# print(f"فایل خروجی ساخته شد: {output_file}")
# print(output.head())




import pandas as pd

# ۱. خواندن فایل فعلی
df = pd.read_csv("GSE98793_MDD_100_probes.csv")

# ۲. اضافه کردن ستون برچسب (چون کل این فایل مربوط به بیماران MDD است)
df['Group'] = 'mdd'  # یا می‌توانی بگذاری 1

# ۳. ذخیره کردن فایل برچسب‌خورده جدید
df.to_csv("GSE98793_MDD_labeled.csv", index=False)

print("فایل جدید با موفقیت برچسب‌گذاری و ذخیره شد!")
print(df['Group'].value_counts())
