import pandas as pd

# ۱. خواندن صحیح فایل CSV
df = pd.read_csv('GPL570-55999.csv'  , low_memory=False)

# اگر باز هم تعداد ستون‌ها ۱ بود، فایل را با جداکننده Tab بخوانید:
if df.shape[1] == 1:
    df = pd.read_csv('GPL570-55999.csv', sep='\t')

# ۲. جستجوی ژن‌ها در سطرها (ستون ID) به جای ستون‌های جدول
selected_genes = ['204493_at', '205404_at', '221780_s_at']
gene_info = df[df['ID'].isin(selected_genes)]

print(gene_info[['ID', 'Gene Symbol', 'Gene Title']])