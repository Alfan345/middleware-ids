#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from app.preprocessing import preprocessor
from app.model import ids_model

ART = Path("artifacts")   # ganti sesuai (atau terima arg)
TEST_CSV = "dataset_contoh.csv"

preprocessor.load()
ids_model.load()

df = pd.read_csv(TEST_CSV, sep=';', engine='python')
df.columns = [c.replace('\ufeff','').strip() for c in df.columns]

# clean thousands separators same as pipeline
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.replace(',', '', regex=False).str.strip().replace({'': None})

expected = preprocessor.get_feature_names()

def show_row(idx):
    raw = df.iloc[idx:idx+1].copy()
    print("\n=== ROW", idx, "raw ===")
    print(raw[expected].iloc[0].to_dict())

    # apply preprocessor recompute and transformations step-by-step
    X_df = raw[expected].copy()
    # convert numeric
    X_df = X_df.apply(pd.to_numeric, errors='coerce')

    # 1) After numeric coercion
    print("\nAfter numeric coercion:")
    print(X_df.iloc[0].to_dict())

    # 2) Recompute rates (call internal function)
    recomputed = preprocessor._recompute_rates(X_df)
    print("\nAfter recompute_rates (show rate cols):")
    for rc in ["Flow Packets/s", "Flow Bytes/s", "Fwd Packets/s", "Bwd Packets/s"]:
        if rc in recomputed.columns:
            print(rc, recomputed[rc].iloc[0])

    # 3) Fill medians
    filled = preprocessor._fill_missing_with_medians(recomputed.copy())
    print("\nAfter fill medians (sample):")
    print({c: filled[c].iloc[0] for c in expected[:8]})

    # 4) Clip
    clipped = preprocessor._apply_clipping(filled.copy())
    print("\nAfter clipping (for top changed features show):")
    changed = []
    for c in expected:
        if filled[c].iloc[0] != clipped[c].iloc[0]:
            changed.append((c, filled[c].iloc[0], clipped[c].iloc[0]))
    print("changed cols:", changed[:20])

    # 5) Log1p on heavy_cols
    rlog = clipped.copy()
    for c in preprocessor.heavy_cols:
        if c in rlog.columns:
            rlog[c] = np.log1p(np.clip(rlog[c], 0, None))
    print("\nAfter log1p (sample heavy cols):")
    for c in preprocessor.heavy_cols[:8]:
        print(c, rlog[c].iloc[0])

    # 6) Scaled vector
    # ensure selection of expected cols
    rfinal = rlog[expected].copy()
    X_scaled = preprocessor.scaler.transform(rfinal.values.astype(float))
    print("\nScaled vector (first 20):", X_scaled.flatten()[:20])

    # 7) Model probs
    preds, probs = ids_model.predict(X_scaled)
    print("\nModel pred/probs:", preds, probs)

# show first N rows
for i in range(min(5, len(df))):
    show_row(i)