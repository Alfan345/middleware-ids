import numpy as np
import pandas as pd

# Ensure preprocessor and artifacts are loaded
from app.preprocessing import preprocessor
preprocessor.load()

# Build `sel` from CSV (semicolon-delimited) and coerce numeric values
df = pd.read_csv("dataset_contoh.csv", sep=';', engine='python')
df.columns = [c.replace('\ufeff','').strip() for c in df.columns]
sel = df[preprocessor.cols].copy()
for c in sel.columns:
    sel[c] = pd.to_numeric(sel[c].astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')

# operate on first row
r = sel.iloc[0:1].copy()
print("raw row:")
print(r.iloc[0,:12])

# fill with medians where NaN
for c in preprocessor.cols:
    if r[c].isna().any():
        r[c] = preprocessor.medians.get(c, r[c].median())

# show heavy_cols effect before log
print("before log (selected heavy cols):")
for c in preprocessor.heavy_cols[:8]:
    print(c, r[c].values)

# apply log1p manually for heavy cols
r_log = r.copy()
for c in preprocessor.heavy_cols:
    if c in r_log.columns:
        r_log[c] = np.log1p(np.clip(r_log[c], 0, None))

print("after log (selected heavy cols):")
for c in preprocessor.heavy_cols[:8]:
    print(c, r_log[c].values)

# apply scaler
X_scaled = preprocessor.scaler.transform(r_log.values.astype(float)).astype('float32')
print("scaled vector (first 20 values):", X_scaled.flatten()[:20])