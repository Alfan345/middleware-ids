#!/usr/bin/env python3
"""
Compute per-feature clip quantiles (e.g. 0.001 and 0.999) from training dataframe
and store them into transform_meta.json under key 'clip_quantiles'.

Usage:
    python scripts/compute_and_store_clip_quantiles.py --artifacts /path/to/artifacts
"""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd


def load_training_df(pkl_path: Path, meta_path: Path):
    data = joblib.load(pkl_path)
    # heuristics
    if isinstance(data, dict):
        if 'full_df' in data:
            return data['full_df']
        if 'X' in data:
            meta = json.loads(meta_path.read_text())
            cols = meta.get('cols')
            import numpy as np
            arr = np.asarray(data['X'])
            if arr.ndim == 2 and cols and arr.shape[1] == len(cols):
                return pd.DataFrame(arr, columns=cols)
            else:
                return pd.DataFrame(arr)
    if isinstance(data, pd.DataFrame):
        return data
    raise RuntimeError("Unable to interpret pkl training artifact structure")


def main(artifacts_dir: Path):
    artifacts_dir = Path(artifacts_dir)
    pkl_path = artifacts_dir / "lite_clean_data_collapsed.pkl"
    meta_path = artifacts_dir / "transform_meta.json"

    if not pkl_path.exists():
        raise SystemExit(f"Training pkl not found: {pkl_path}")
    if not meta_path.exists():
        raise SystemExit(f"transform_meta.json not found: {meta_path}")

    print("Loading training artifact...")
    df_train = load_training_df(pkl_path, meta_path)
    meta = json.loads(meta_path.read_text())
    cols = meta.get("cols", [])

    clip = {}
    for c in cols:
        if c in df_train.columns:
            ser = pd.to_numeric(df_train[c], errors='coerce').dropna()
            if len(ser) > 0:
                low = float(ser.quantile(0.001))
                high = float(ser.quantile(0.999))
                clip[c] = [low, high]
            else:
                clip[c] = [None, None]
        else:
            clip[c] = [None, None]

    meta['clip_quantiles'] = clip
    # backup
    backup = meta_path.with_suffix('.json.clipbak')
    backup.write_text(meta_path.read_text())
    meta_path.write_text(json.dumps(meta, indent=2))
    print("Wrote clip_quantiles to", meta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', '-a', default='artifacts', help='Artifacts directory')
    args = parser.parse_args()
    main(Path(args.artifacts))