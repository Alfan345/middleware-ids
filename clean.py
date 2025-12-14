#!/usr/bin/env python3
"""
Clean CSV (handle thousands separators / decimal comma), save cleaned CSV,
and run local inference using app.preprocessor + app.ids_model for quick feedback.

Usage:
  python scripts/clean_and_predict.py --input dataset_contoh.csv --artifacts artifacts
"""
import argparse, io, sys, json
from pathlib import Path
import pandas as pd, numpy as np

def try_read_with_thousands(path):
    # try separators and choose best numeric parse
    candidates = [',',';','\t']
    best = None
    best_score = -1
    last_err = None
    raw = Path(path).read_bytes()
    for sep in candidates:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, thousands=',', quotechar='"', engine='python')
            # score = total numeric entries
            score = 0
            for c in df.columns:
                coerced = pd.to_numeric(df[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
                score += coerced.notna().sum()
            if score > best_score:
                best_score = score
                best = (df, sep)
        except Exception as e:
            last_err = e
    return best, last_err

def smart_fix_commas(df, train_medians=None):
    report = {}
    cols = [c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.contains(',').any()]
    for c in cols:
        s = df[c].astype(str).fillna('')
        tmpA = s.str.replace(',', '', regex=False).replace({'': None})
        numA = pd.to_numeric(tmpA, errors='coerce')
        tmpB = s.str.replace(',', '.', regex=False).replace({'': None})
        numB = pd.to_numeric(tmpB, errors='coerce')
        naA = int(numA.isna().sum()); naB = int(numB.isna().sum())
        chosen = None; reason = None
        if train_medians and c in train_medians and train_medians[c] is not None:
            train_med = float(train_medians[c])
            def rel_diff(x):
                if x is None or (isinstance(x, float) and np.isnan(x)): return np.inf
                if train_med == 0:
                    return abs(float(x) - train_med)
                return abs((float(x) - train_med) / (train_med + 1e-12))
            medA = numA.median(skipna=True); medB = numB.median(skipna=True)
            dA = rel_diff(medA); dB = rel_diff(medB)
            if dA < dB:
                chosen = numA; reason = f"closer to train median (dA={dA:.3g} < dB={dB:.3g})"
            else:
                chosen = numB; reason = f"closer to train median (dB={dB:.3g} <= dA={dA:.3g})"
        else:
            if naA < naB:
                chosen = numA; reason = f"fewer NaNs ({naA} < {naB})"
            elif naB < naA:
                chosen = numB; reason = f"fewer NaNs ({naB} < {naA})"
            else:
                medA = numA.median(skipna=True) if naA < len(numA) else np.nan
                medB = numB.median(skipna=True) if naB < len(numB) else np.nan
                if abs(medA if not np.isnan(medA) else np.inf) <= abs(medB if not np.isnan(medB) else np.inf):
                    chosen = numA; reason = "tie -> smaller median magnitude (A)"
                else:
                    chosen = numB; reason = "tie -> smaller median magnitude (B)"
        if chosen is None:
            chosen = numA; reason = "fallback remove commas"
        df[c] = chosen
        report[c] = {"na_after": int(chosen.isna().sum()), "median_after": None if chosen.dropna().empty else float(chosen.median()), "choice": reason}
    return df, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i', required=True)
    parser.add_argument('--artifacts','-a', default='artifacts')
    args = parser.parse_args()

    # import app pieces (repo root must be CWD or package installed)
    try:
        from app.preprocessing import preprocessor
        from app.model import ids_model
    except Exception as e:
        print("Error importing app modules:", e); sys.exit(1)

    artifacts = Path(args.artifacts)
    preprocessor.load(artifacts_dir=artifacts)

    print("Trying read CSV with thousands=',' ...")
    (df, sep), err = try_read_with_thousands(args.input)
    if df is None:
        print("Failed read with heuristics, abort. Err:", err); sys.exit(1)
    print("Selected separator:", sep)
    # normalize columns
    df.columns = [c.replace('\ufeff','').strip() for c in df.columns]

    # quick check for object cols containing comma
    obj_has_comma = [c for c in df.columns if df[c].dtype==object and df[c].astype(str).str.contains(',').any()]
    print("Object columns containing comma before smart-fix:", obj_has_comma)

    # smart fix using training medians
    df_fixed, report = smart_fix_commas(df, train_medians=preprocessor.medians)
    if report:
        print("Smart-fix report:", json.dumps(report, indent=2))

    # coerce all to numeric where possible (except Label)
    if 'Label' in df_fixed.columns:
        lab = df_fixed['Label'].copy()
    else:
        lab = None
    for c in df_fixed.columns:
        if c == 'Label': continue
        df_fixed[c] = pd.to_numeric(df_fixed[c], errors='coerce')

    # show summaries for problematic features
    suspects = ["Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Mean","Flow Packets/s","Bwd Packets/s","Packet Length Mean"]
    print("\nColumn stats (min, median, max) for suspects:")
    for c in suspects:
        if c in df_fixed.columns:
            ser = df_fixed[c].dropna()
            print(c, "min:", None if ser.empty else float(ser.min()), "median:", None if ser.empty else float(ser.median()), "max:", None if ser.empty else float(ser.max()))

    out = Path(Path(args.input).stem + "_cleaned.csv")
    # if Label existed, restore it
    if lab is not None:
        df_fixed['Label'] = lab
    df_fixed.to_csv(out, index=False)
    print("Wrote cleaned CSV to", out)

    # now run local inference (batch)
    # prepare features expected by preprocessor
    features = preprocessor.get_feature_names()
    missing = [c for c in features if c not in df_fixed.columns]
    if missing:
        print("Missing features required by preprocessor:", missing)
        print("Aborting inference.")
        sys.exit(1)

    X = preprocessor.transform(df_fixed[features])
    ids_model.load(artifacts_dir=artifacts)
    results = ids_model.predict_batch(X)
    # summarize results
    from collections import Counter
    preds = [r['prediction'] for r in results]
    cnt = Counter(preds)
    print("\nPrediction counts:", dict(cnt))
    print("\nSample results (first 10):")
    for r in results[:10]:
        print(r)

if __name__ == '__main__':
    main()