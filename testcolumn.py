# cek fitur yang diharapkan vs header CSV
import pandas as pd
from app.preprocessing import preprocessor
from app.column_mapper import SimpleColumnMapper

# load preprocessor (agar preprocessor.get_feature_names tersedia)
preprocessor.load()   # pastikan artifacts sudah ada
expected = preprocessor.get_feature_names()
print("Expected count:", len(expected))
print(expected)

# load csv dan normalisasi header sama seperti kode Anda
# dataset_contoh.csv uses semicolon separators — read with sep=';'
df = pd.read_csv("dataset_contoh.csv", sep=';', engine='python')
df.columns = [c.replace('\ufeff','').strip() for c in df.columns]
print("Columns in CSV (count):", len(df.columns))
print(df.columns.tolist())

# cek duplikat
dups = [c for c, n in pd.Series(df.columns).value_counts().items() if n>1]
print("Duplicate column names:", dups)

# mapping
mapped_df, mapping = SimpleColumnMapper.map_columns(df, expected)
print("Mapping applied (sample):", list(mapping.items())[:30])

# cek missing setelah mapping
missing_after_map = [c for c in expected if c not in mapped_df.columns]
print("Missing after mapping:", missing_after_map)

# tunjukkan sample data yang akan masuk model (jika tidak missing)
if not missing_after_map:
    df_selected = mapped_df[expected].copy()
    print("Selected head:")
    print(df_selected.head(3))
    print("NaN counts:", df_selected.isna().sum().loc[lambda s: s>0])