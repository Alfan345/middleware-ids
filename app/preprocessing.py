"""
Preprocessing pipeline for flow features
Matches the transform pipeline used during training

Enhancements:
- Loads optional 'clip_quantiles' from transform_meta.json (per-feature [q001, q999]).
- Recomputes rate features (Flow Packets/s, Flow Bytes/s, Fwd/Bwd Packets/s) from counts + Flow Duration
  with a heuristic to detect duration unit (seconds / ms / microseconds).
- Clips features to training quantile bounds before log1p and scaling to reduce out-of-distribution extremes.
- Conservative clipping: if many rows would be clipped for a feature, set extremes to NaN (so they'll be median-filled)
  and record a clip report for monitoring.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union

from app.config import settings


class FlowPreprocessor:
    """Preprocessor for network flow features"""

    def __init__(self):
        self.scaler = None
        self.cols = None
        self.heavy_cols = None
        self.medians = None
        self.clip_map = {}           # feature -> [low, high]
        self.is_loaded = False
        self._has_medians = False
        self._last_clip_report = {}

    def load(self, artifacts_dir: Path = None):
        """Load preprocessing artifacts"""
        if artifacts_dir is None:
            artifacts_dir = settings.ARTIFACTS_DIR

        # Load scaler
        scaler_path = artifacts_dir / settings.SCALER_FILE
        self.scaler = joblib.load(scaler_path)

        # Load transform metadata
        meta_path = artifacts_dir / settings.TRANSFORM_META_FILE
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.cols = meta['cols']
        self.heavy_cols = meta.get('heavy_cols', [])
        self.medians = meta.get('medians', {}) or {}
        self.clip_map = meta.get('clip_quantiles', {}) or {}

        # If medians is empty, try to extract from scaler (RobustScaler.center_ / median_)
        if not self.medians:
            center = getattr(self.scaler, 'center_', None) or getattr(self.scaler, 'median_', None)
            if center is not None and len(center) == len(self.cols):
                self.medians = {c: float(center[i]) for i, c in enumerate(self.cols)}
                print("Preprocessor: filled medians from scaler center_/median_")

        self._has_medians = bool(self.medians)
        if not self._has_medians:
            print("Warning: no medians found in transform_meta.json and scaler; runtime NaN filling will be disabled.")

        self.is_loaded = True
        print(f"✅ Preprocessor loaded: {len(self.cols)} features")

    def _guess_duration_unit_and_to_seconds(self, duration_ser: pd.Series, train_median: float = None) -> pd.Series:
        """
        Heuristic to convert Flow Duration to seconds:
        - If training median duration is much smaller than test median, try to scale accordingly.
        - Typical units: seconds, milliseconds (ms), microseconds (us).
        """
        dur = pd.to_numeric(duration_ser, errors='coerce').copy()

        # simple heuristics using magnitudes
        median_test = float(dur.median()) if not dur.dropna().empty else None
        median_train = float(train_median) if train_median is not None else None

        # If values are tiny or unavailable, assume already seconds
        if median_test is None:
            return dur

        # If train median is available and test median >> train median (by >1000), attempt to scale down:
        if median_train is not None and median_test / (median_train + 1e-12) > 1000:
            # assume test durations in microseconds or milliseconds; choose factor accordingly
            if median_test > 1e6:
                # microseconds -> seconds
                return dur / 1e6
            else:
                # milliseconds -> seconds
                return dur / 1e3

        # If test median extremely large (>1e6) assume microseconds
        if median_test > 1e6:
            return dur / 1e6
        # If test median moderately large (>1e3) assume milliseconds
        if median_test > 1e3:
            return dur / 1e3

        # otherwise assume seconds
        return dur

    def _recompute_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # required column names (best-effort)
        dur_col = None
        for cand in ["Flow Duration", "flow duration", "Duration"]:
            if cand in df.columns:
                dur_col = cand
                break
        if dur_col is None:
            return df

        # ensure numeric
        df[dur_col] = pd.to_numeric(df[dur_col], errors='coerce')

        # get training median for duration if available
        train_dur_med = self.medians.get("Flow Duration") if self.medians else None

        # convert to seconds using heuristic
        dur_s = self._guess_duration_unit_and_to_seconds(df[dur_col], train_dur_med)

        # helper to get column as numeric Series (or zeros if missing)
        def as_series(col_name):
            if col_name in df.columns:
                return pd.to_numeric(df[col_name], errors='coerce').fillna(0)
            else:
                return pd.Series(0.0, index=df.index, dtype=float)

        fwd_pk = as_series("Total Fwd Packets")
        bwd_pk = as_series("Total Backward Packets")
        total_pk = fwd_pk + bwd_pk

        fwd_bytes = as_series("Total Length of Fwd Packets")
        bwd_bytes = as_series("Total Length of Bwd Packets")
        total_bytes = fwd_bytes + bwd_bytes

        eps = 1e-6
        dur_s_safe = dur_s.replace(0, np.nan).fillna(eps)

        # recompute and assign (override existing rate cols if present)
        df["Flow Packets/s"] = (total_pk / dur_s_safe).replace([np.inf, -np.inf], np.nan)
        df["Flow Bytes/s"] = (total_bytes / dur_s_safe).replace([np.inf, -np.inf], np.nan)
        df["Fwd Packets/s"] = (fwd_pk / dur_s_safe).replace([np.inf, -np.inf], np.nan)
        df["Bwd Packets/s"] = (bwd_pk / dur_s_safe).replace([np.inf, -np.inf], np.nan)

        # For rows with non-positive duration, set rates to 0 (consistent policy)
        mask_nonpos = dur_s <= 0
        if mask_nonpos.any():
            df.loc[mask_nonpos, ["Flow Packets/s", "Flow Bytes/s", "Fwd Packets/s", "Bwd Packets/s"]] = 0.0

        return df

    def _fill_missing_with_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in df using stored training medians.

        Assumes df columns are already subset/ordered to self.cols.
        """
        if not self._has_medians:
            return df
        for c in self.cols:
            if c in df.columns:
                if df[c].isna().any():
                    fill_val = self.medians.get(c, df[c].median())
                    df[c] = df[c].fillna(fill_val)
        return df

    def _apply_clipping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Conservative clipping:
         - For each feature with bounds, compute how many rows would be clipped.
         - If >1% of rows would be clipped for that feature, set extreme values to NaN
           (so they will be median-filled), and record in clip report.
         - Otherwise apply clipping to keep values within bounds.
        """
        self._last_clip_report = {}
        if not self.clip_map:
            return df
        total = len(df)
        for c, bounds in self.clip_map.items():
            if c not in df.columns or not bounds or bounds[0] is None or bounds[1] is None:
                continue
            try:
                low, high = float(bounds[0]), float(bounds[1])
            except Exception:
                continue
            below = (df[c] < low).sum()
            above = (df[c] > high).sum()
            frac = (below + above) / max(1, total)
            self._last_clip_report[c] = {"below": int(below), "above": int(above), "frac": float(frac)}
            # if many rows affected -> mark suspicious and set extremes to NaN for safer median imputation
            if frac > 0.01:  # threshold tunable
                mask = (df[c] < low) | (df[c] > high)
                df.loc[mask, c] = np.nan
            else:
                df[c] = df[c].clip(lower=low, upper=high)
        return df

    def transform(self, features: Union[Dict[str, float], pd.DataFrame]) -> np.ndarray:
        """
        Transform input features using the fitted pipeline

        Args:
            features: Dictionary or DataFrame of flow features

        Returns:
            Transformed numpy array ready for model inference
        """
        if not self.is_loaded:
            raise RuntimeError("Preprocessor not loaded.  Call load() first.")

        # Convert to DataFrame if dict
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        # Ensure all expected columns exist (create NaN for missing)
        for c in self.cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[self.cols].copy()

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Recompute rates robustly (overrides uploaded precomputed rates)
        df = self._recompute_rates(df)

        # Fill missing values: use medians (if available); else error out
        if df.isna().any().any():
            if self._has_medians:
                df = self._fill_missing_with_medians(df)
            else:
                missing_cols = [c for c in self.cols if df[c].isna().any()]
                raise ValueError(
                    "Input contains NaN for columns but no training medians are available. "
                    "Run scripts/compute_and_store_clip_quantiles.py to populate transform_meta.json or provide complete data. "
                    f"Problem columns: {missing_cols}"
                )

                # Apply conservative clipping using training quantiles (if available)
        df = self._apply_clipping(df)

        # IMPORTANT: clipping may have created NaNs (we replaced extremes with NaN on purpose).
        # Fill those NaNs again with training medians to avoid NaNs propagating to scaler/model.
        if df.isna().any().any():
            if self._has_medians:
                df = self._fill_missing_with_medians(df)
            else:
                # as a fallback, replace remaining NaNs with 0 to avoid crashing model (less ideal)
                df = df.fillna(0)

        # Apply log transform to heavy-tailed columns
        for c in self.heavy_cols:
            if c in df.columns:
                df[c] = np.log1p(np.clip(df[c], 0, None))

        # Ensure we only pass exactly the expected columns (drop any extras added by recompute)
        df = df[self.cols].copy()

        # Final safety: replace any non-finite values
        arr = df.values.astype(float)
        if not np.isfinite(arr).all():
            # replace inf with large finite numbers and nan with medians already attempted above
            arr = np.where(np.isfinite(arr), arr, np.nan)
            # if still NaN, fill with 0
            arr = np.nan_to_num(arr, nan=0.0, posinf=np.finfo(np.float32).max, neginf=-np.finfo(np.float32).max)

        # Scale features
        X_scaled = self.scaler.transform(arr).astype('float32')
        return X_scaled

    def transform_batch(self, flows: List[Dict[str, float]]) -> np.ndarray:
        """Transform a batch of flow features"""
        df = pd.DataFrame(flows)
        return self.transform(df)

    def get_feature_names(self) -> List[str]:
        """Get list of required feature names"""
        return self.cols.copy()

    def validate_features(self, features: Dict[str, float]) -> tuple:
        """
        Validate that all required features are present

        Returns:
            (is_valid, missing_features)
        """
        missing = [col for col in self.cols if col not in features]
        return len(missing) == 0, missing

    def get_last_clip_report(self) -> Dict:
        """Return the last clip report produced by transform()"""
        return getattr(self, "_last_clip_report", {})


# Global preprocessor instance
preprocessor = FlowPreprocessor()