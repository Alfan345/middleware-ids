import pandas as pd

# Ensure preprocessor is loaded
from app.preprocessing import preprocessor
preprocessor.load()

# dataset_contoh.csv is semicolon-delimited and contains thousands separators (commas)
df = pd.read_csv("dataset_contoh.csv", sep=';', engine='python')
df.columns = [c.replace('\ufeff','').strip() for c in df.columns]

# Select required cols and coerce to numeric (remove thousand separators)
sel = df[preprocessor.cols].copy()
for c in sel.columns:
    # replace commas used as thousands separators, strip spaces, then convert
    sel[c] = pd.to_numeric(sel[c].astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')

# raw stats
print("CSV sample stats (first 10 cols):")
print(sel.describe().loc[['min','50%','max']].iloc[:10])

# compare medians
for c in preprocessor.cols[:10]:
    print(c, "meta_median:", preprocessor.medians.get(c), "csv_median:", sel[c].median())