#!/usr/bin/env python3
"""
Diagnosis script for preprocessing + inference pipeline

What this script does:
- Loads preprocessing artifacts and model via the middleware app instances (preprocessor & ids_model).
- Loads training artifact (lite_clean_data_collapsed.pkl) to get the "raw" training DataFrame (full_df).
- Loads the test CSV you provide, normalizes headers and numeric formats (handles thousands separators).
- Runs several checks and produces a JSON + human-readable summary:
  1) Sanity-check prediction on a sample of training rows (predict with current pipeline and compare to actual labels).
  2) Feature distribution drift checks (KS test) between training sample and provided test CSV.
  3) Compare medians (training vs test) and list top differences.
  4) Report on rate/duration extremes and number of rows with Flow Duration <= 0.
  5) Show example test rows with extreme values for problematic features.
- Writes a report file (diagnose_report.json) in the current directory.

Usage:
    python scripts/diagnose_pipeline.py --artifacts /path/to/artifacts --test_csv /path/to/dataset_contoh.csv

Notes:
- The script expects the middleware repo (app package) is importable (run from repo root or install package).
- It will attempt to load lite_clean_data_collapsed.pkl and transform_meta.json from --artifacts.
- If scipy is not available, KS tests will be skipped with a warning.

"""
import argparse
import json
import math
import os
import random
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

# Try optional dependency
try:
    from scipy.stats import ks_2samp
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Import app objects (assumes running from repo root or package installed)
try:
    from app.preprocessing import preprocessor
    from app.model import ids_model
    from app.config import settings
except Exception as e:
    raise SystemExit("Failed to import app.* modules. Run this from the repository root or ensure package is importable. Error: %s" % e)


def read_test_csv(path):
    """
    Read CSV/Excel/JSON with tolerant parsing:
    - Try pandas read_csv with common separators, handle thousands separators in object columns.
    - Normalize headers (strip + remove BOM).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # Try multiple readers for common separators
    tried = {}
    df = None
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif p.suffix.lower() in (".json",):
        df = pd.read_json(p)
    else:
        sep_candidates = [",", ";", "\t"]
        last_exc = None
        for sep in sep_candidates:
            try:
                df = pd.read_csv(p, sep=sep, engine='python')
                # basic sanity: expect header length >= 10
                if df.shape[1] >= 10:
                    break
            except Exception as e:
                last_exc = e
        if df is None:
            # fallback to default read_csv
            df = pd.read_csv(p, engine='python')

    # normalize headers
    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]

    # Drop duplicate column names (keep first)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Heuristic: detect object columns that likely contain thousands separators like "1,234,567"
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    thousands_re = r'^\s*[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*$'
    cols_fixed = []
    for c in obj_cols:
        sample = df[c].dropna().astype(str).head(30).tolist()
        if not sample:
            continue
        if any(pd.Series(sample).str.match(thousands_re).fillna(False)):
            # remove commas and convert
            try:
                df[c] = df[c].astype(str).str.replace(',', '', regex=False).str.strip().replace({'': None})
                df[c] = pd.to_numeric(df[c], errors='coerce')
                cols_fixed.append(c)
            except Exception:
                pass

    return df, cols_fixed


def load_training_df(artifacts_dir: Path):
    pkl_path = artifacts_dir / "lite_clean_data_collapsed.pkl"
    if not pkl_path.exists():
        return None, "pkl not found"
    try:
        data = joblib.load(pkl_path)
    except Exception as e:
        return None, f"failed to unpickle pkl: {e}"

    # heuristics to retrieve DataFrame
    if isinstance(data, dict):
        if 'full_df' in data and data['full_df'] is not None:
            return data['full_df'], None
        # try 'X' with 'cols' in transform_meta.json
        if 'X' in data and isinstance(data['X'], (list, np.ndarray)):
            # try to build DataFrame using transform_meta.json cols
            meta_path = artifacts_dir / "transform_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                cols = meta.get('cols')
                if cols and len(cols) == np.asarray(data['X']).shape[1]:
                    return pd.DataFrame(data['X'], columns=cols), None
            # fallback DataFrame from X without columns
            return pd.DataFrame(data['X']), None
    # fallback: if data itself is DataFrame
    if isinstance(data, pd.DataFrame):
        return data, None
    return None, "unknown pkl structure"


def safe_to_numeric_df(df, cols):
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            continue
        # remove thousands separators if object
        if df2[c].dtype == object:
            df2[c] = df2[c].astype(str).str.replace(',', '', regex=False).str.strip().replace({'': None})
        df2[c] = pd.to_numeric(df2[c], errors='coerce')
    return df2


def sample_predict_train(df_train, n=500, random_state=42):
    sample = df_train.sample(n=min(n, len(df_train)), random_state=random_state)
    # If there's a Label column name different, try common names
    label_col = None
    for cand in ["Label", "label", "label_col", "label_text"]:
        if cand in sample.columns:
            label_col = cand
            break
    if label_col is None:
        # try finding a column that matches id_to_label values?
        label_col = 'Label'  # best-effort, may fail

    X_df = sample[preprocessor.get_feature_names()]
    # ensure numeric
    X_df = safe_to_numeric_df(X_df, X_df.columns)
    X = preprocessor.transform(X_df)
    preds = ids_model.predict_batch(X)
    pred_labels = [p['prediction'] for p in preds]
    actual_labels = list(sample[label_col].astype(str).values) if label_col in sample.columns else [None]*len(pred_labels)

    # compute simple accuracy (only for rows with actual label)
    matches = [1 if a is not None and p == a else 0 for p, a in zip(pred_labels, actual_labels)]
    acc = sum(matches) / len(matches) if sum(1 for a in actual_labels if a is not None) > 0 else None

    # collect mismatches (up to 20)
    mismatches = []
    for i, (row_idx) in enumerate(sample.index):
        act = actual_labels[i] if i < len(actual_labels) else None
        pred = pred_labels[i]
        if act is not None and pred != act:
            mismatches.append({
                "index": int(row_idx),
                "actual": act,
                "predicted": pred,
            })
            if len(mismatches) >= 20:
                break

    return {
        "sample_n": len(pred_labels),
        "label_col": label_col,
        "accuracy": acc,
        "mismatches_count": len(mismatches),
        "mismatches": mismatches,
        "predicted_sample": pred_labels[:10],
        "actual_sample": actual_labels[:10]
    }


def ks_tests(train_ser, test_ser, nsamples=2000):
    if not HAS_SCIPY:
        return None, "scipy not installed"
    # sample without replacement
    t = train_ser.dropna()
    s = test_ser.dropna()
    if len(t) < 20 or len(s) < 20:
        return None, "not enough data"
    ta = t.sample(n=min(nsamples, len(t)), random_state=1).values
    sa = s.sample(n=min(nsamples, len(s)), random_state=1).values
    res = ks_2samp(ta, sa)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}, None


def compute_report(artifacts, test_csv):
    artifacts = Path(artifacts)
    report = {"artifacts": str(artifacts), "test_csv": str(test_csv)}

    # 1) load meta and preprocessor & model
    try:
        preprocessor.load(artifacts_dir=artifacts)
    except Exception as e:
        # try load() without arg
        try:
            preprocessor.load()
        except Exception:
            report["error_preprocessor_load"] = str(e)

    try:
        ids_model.load(artifacts_dir=artifacts)
    except Exception as e:
        try:
            ids_model.load()
        except Exception:
            report["error_model_load"] = str(e)

    report["preprocessor_loaded"] = preprocessor.is_loaded
    report["model_loaded"] = ids_model.is_loaded

    # 2) load training df from pkl
    df_train, train_err = load_training_df(artifacts)
    report["train_load_error"] = train_err
    if df_train is None:
        report["note"] = "Training dataframe not available; some checks skipped"
    else:
        report["train_shape"] = df_train.shape
        # make sure label column exists if possible
        if 'Label' in df_train.columns:
            report["train_label_counts"] = df_train['Label'].value_counts().to_dict()

    # 3) load test csv and clean numeric formats
    df_test, cleaned_cols = read_test_csv(test_csv)
    report["test_shape"] = df_test.shape
    report["cleaned_columns_thousands_sep"] = cleaned_cols

    # 4) Sanity-check prediction on training sample (if available)
    if df_train is not None:
        try:
            sc_pred = sample_predict_train(df_train, n=500)
            report["sanity_train_prediction"] = sc_pred
        except Exception as e:
            report["sanity_train_prediction_error"] = str(e)

    # 5) Compare medians (train vs test) on the features expected by preprocessor
    expected = preprocessor.get_feature_names()
    med_compare = {}
    top_diff = []
    for c in expected:
        train_med = None
        test_med = None
        try:
            if df_train is not None and c in df_train.columns:
                train_med = float(pd.to_numeric(df_train[c], errors='coerce').median(skipna=True))
            if c in df_test.columns:
                test_med = float(pd.to_numeric(df_test[c], errors='coerce').median(skipna=True))
        except Exception:
            pass
        med_compare[c] = {"train_median": train_med, "test_median": test_med}
        if train_med is not None and test_med is not None:
            # compute ratio/absolute difference
            if train_med == 0:
                diff = abs(test_med - train_med)
            else:
                diff = abs((test_med - train_med) / float(train_med))
            top_diff.append((c, diff, train_med, test_med))
    top_diff.sort(key=lambda x: x[1], reverse=True)
    report["median_comparison"] = med_compare
    report["top_median_shifts"] = [{"feature": f, "rel_diff": float(d), "train_med": tm, "test_med": ts} for f, d, tm, ts in top_diff[:30]]

    # 6) KS tests (if scipy available)
    ks_results = {}
    if HAS_SCIPY:
        for c in expected:
            if c in df_test.columns and df_train is not None and c in df_train.columns:
                try:
                    res, err = ks_tests(df_train[c], df_test[c])
                    ks_results[c] = res
                except Exception as e:
                    ks_results[c] = {"error": str(e)}
    else:
        ks_results = {"skipped": "scipy not installed; install scipy to run KS tests (pip install scipy)"}
    report["ks_tests"] = ks_results

    # 7) Duration/rates extremes
    duration_col = None
    for cand in ["Flow Duration", "flow duration", "Duration"]:
        if cand in df_test.columns:
            duration_col = cand
            break
    rate_cols = [c for c in expected if 'Flow Bytes/s' in c or 'Flow Packets/s' in c or 'Packets/s' in c or 'Packets/s' in c or 'Bwd Packets/s' in c]
    rate_cols = [c for c in rate_cols if c in df_test.columns]

    report["duration_col"] = duration_col
    if duration_col:
        df_test_num = safe_to_numeric_df(df_test, [duration_col])
        report["test_duration_le_zero"] = int((df_test_num[duration_col] <= 0).sum())
        report["test_duration_min"] = float(df_test_num[duration_col].min())
        report["test_duration_median"] = float(df_test_num[duration_col].median())
        report["test_duration_max"] = float(df_test_num[duration_col].max())

    rate_report = {}
    for c in rate_cols:
        ser = pd.to_numeric(df_test[c], errors='coerce').dropna()
        if df_train is not None and c in df_train.columns:
            train_q999 = float(pd.to_numeric(df_train[c], errors='coerce').quantile(0.999))
        else:
            train_q999 = None
        over_q999 = int((ser > train_q999).sum()) if train_q999 is not None else None
        rate_report[c] = {"test_min": float(ser.min()) if not ser.empty else None,
                          "test_median": float(ser.median()) if not ser.empty else None,
                          "test_max": float(ser.max()) if not ser.empty else None,
                          "train_q999": train_q999,
                          "test_over_train_q999": over_q999}
    report["rate_report"] = rate_report

    # 8) Example extreme rows (top 5 by max normalized zscore vs train median/scale)
    extremes = []
    if df_train is not None:
        # build robust stats from train medians and scale_ if available
        train_stats = {}
        for c in expected:
            tser = pd.to_numeric(df_train[c], errors='coerce').dropna()
            if len(tser) > 0:
                train_stats[c] = {"median": float(tser.median()), "q99": float(tser.quantile(0.99)), "q999": float(tser.quantile(0.999))}
            else:
                train_stats[c] = {"median": None, "q99": None, "q999": None}

        # compute a simple score per row in test: max(abs((val - train_median)/train_median))
        row_scores = []
        df_test_num = safe_to_numeric_df(df_test, expected)
        for idx, row in df_test_num.iterrows():
            max_score = 0.0
            for c in expected:
                if c not in df_test_num.columns:
                    continue
                v = row[c]
                m = train_stats[c]["median"]
                if v is None or pd.isna(v) or m is None or m == 0:
                    continue
                score = abs((v - m) / float(m))
                if math.isfinite(score) and score > max_score:
                    max_score = score
            row_scores.append((idx, max_score))
        row_scores.sort(key=lambda x: x[1], reverse=True)
        for idx, score in row_scores[:10]:
            extremes.append({"index": int(idx), "score": float(score), "row": df_test.loc[idx, expected].to_dict()})
    report["extreme_rows"] = extremes[:10]

    # 9) Basic mapping check
    mapped_df, mapping = None, None
    try:
        from app.column_mapper import SimpleColumnMapper
        mapped_df, mapping = SimpleColumnMapper.map_columns(df_test, expected)
        report["mapping_applied"] = mapping
        report["missing_after_map"] = [c for c in expected if c not in mapped_df.columns]
    except Exception as e:
        report["mapping_error"] = str(e)

    # Save report
    out_path = Path("diagnose_report.json")
    out_path.write_text(json.dumps(report, indent=2, default=lambda x: str(x)))
    return report


def main():
    parser = argparse.ArgumentParser(description="Diagnose preprocessing + model pipeline")
    parser.add_argument("--artifacts", "-a", required=True, help="Artifacts directory path (where transform_meta.json, scaler.pkl, lite_clean_data_collapsed.pkl live)")
    parser.add_argument("--test_csv", "-t", required=True, help="Path to test CSV (dataset to diagnose)")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    test_csv = Path(args.test_csv)

    print("Running diagnosis...")
    report = compute_report(artifacts, test_csv)

    # Print summary
    print("\n=== Summary ===")
    print("Artifacts:", report.get("artifacts"))
    print("Test CSV:", report.get("test_csv"))
    print("Preprocessor loaded:", report.get("preprocessor_loaded"))
    print("Model loaded:", report.get("model_loaded"))
    if "sanity_train_prediction" in report:
        sp = report["sanity_train_prediction"]
        print("Sanity-check on train sample:", sp.get("sample_n"), "rows")
        print(" - label col:", sp.get("label_col"))
        print(" - accuracy on sample:", sp.get("accuracy"))
        print(" - mismatches:", sp.get("mismatches_count"))
    if report.get("top_median_shifts"):
        print("\nTop median shifts (feature, rel_diff, train_med, test_med):")
        for rec in report["top_median_shifts"][:10]:
            print(" ", rec["feature"], round(rec["rel_diff"], 3), rec["train_med"], rec["test_med"])
    if "ks_tests" in report and isinstance(report["ks_tests"], dict) and report["ks_tests"]:
        print("\nKS tests (showing features with pvalue < 0.001 if computed):")
        if not HAS_SCIPY:
            print("  scipy not installed; KS tests skipped")
        else:
            small_p = {f: v for f, v in report["ks_tests"].items() if isinstance(v, dict) and v.get("pvalue") is not None and v["pvalue"] < 0.001}
            if small_p:
                for f, v in small_p.items():
                    print(" ", f, "stat:", v.get("statistic"), "p:", v.get("pvalue"))
            else:
                print("  no strong KS rejects (p<0.001) among compared features")

    print("\nDetailed JSON report saved to diagnose_report.json")
    print("Done.")

if __name__ == "__main__":
    main()