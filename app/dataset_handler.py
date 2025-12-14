"""
Dataset handler for file upload and batch prediction
"""
import io
import time
import re
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from fastapi import UploadFile, HTTPException

from app.preprocessing import preprocessor
from app.model import ids_model
from app.column_mapper import SimpleColumnMapper


# Helper: smart comma fixer (tries remove-thousands OR comma->dot)
def _smart_fix_commas(df: pd.DataFrame, train_medians: dict = None, sample_n: int = 100):
    """
    For each object column that contains commas, try two fixes:
      A) Remove commas (thousands separators): '1,234' -> '1234'
      B) Replace comma with dot (decimal comma): '1,234' -> '1.234'
    Choose the fix whose resulting numeric median is closest to training median (if provided).
    If no training median available, choose the option that yields fewer NaNs and smaller magnitude.
    Returns: df_fixed, report dict
    """
    report = {}
    cols = [c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.contains(',').any()]
    for c in cols:
        s = df[c].astype(str).fillna('')
        sample = s[s != ''].head(sample_n).tolist()
        if not sample:
            continue

        # Candidate A: remove commas
        tmpA = s.str.replace(',', '', regex=False).replace({'': None})
        numA = pd.to_numeric(tmpA, errors='coerce')

        # Candidate B: comma -> dot
        tmpB = s.str.replace(',', '.', regex=False).replace({'': None})
        numB = pd.to_numeric(tmpB, errors='coerce')

        naA = int(numA.isna().sum())
        naB = int(numB.isna().sum())

        chosen = None
        reason = None

        if train_medians and c in train_medians and train_medians[c] is not None:
            train_med = float(train_medians[c])
            def rel_diff(x):
                if x is None or (isinstance(x, float) and np.isnan(x)): return np.inf
                if train_med == 0:
                    return abs(float(x) - train_med)
                return abs((float(x) - train_med) / (train_med + 1e-12))
            medA = numA.median(skipna=True)
            medB = numB.median(skipna=True)
            dA = rel_diff(medA); dB = rel_diff(medB)
            if dA < dB:
                chosen = numA; reason = f"closer to train median (dA={dA:.3g} < dB={dB:.3g})"
            else:
                chosen = numB; reason = f"closer to train median (dB={dB:.3g} <= dA={dA:.3g})"
        else:
            # fallback: prefer fewer NaNs
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
            chosen = numA
            reason = "fallback remove commas"

        df[c] = chosen
        report[c] = {
            "na_after": int(chosen.isna().sum()),
            "median_after": None if chosen.dropna().empty else float(chosen.median()),
            "choice_reason": reason
        }
    return df, report


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object so it is JSON serializable:
    - Convert numpy scalars to Python scalars
    - Replace NaN / inf with None
    - Convert numpy arrays to lists
    """
    # dict
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # numpy array
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    # numpy scalar
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    # python float/int
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    # pandas types
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass
    # fallback to string
    try:
        return str(obj)
    except Exception:
        return None


class DatasetHandler:
    """Handle dataset upload and prediction"""

    SUPPORTED_FORMATS = ['.csv', '.json', '.xlsx']
    MAX_FILE_SIZE_MB = 100

    def __init__(self):
        self.last_dataset: Optional[pd.DataFrame] = None
        self.last_results: Optional[List[Dict]] = None
        self.last_mapping: Optional[Dict] = None
        self.last_summary: Optional[Dict] = None

    async def read_file(self, file: UploadFile) -> pd.DataFrame:
        """Read uploaded file into DataFrame and normalize numeric formats.

        This function tries multiple separators and uses pandas' thousands argument to
        correctly parse numbers with thousands separators. It picks the parse that
        results in the most numeric columns (heuristic).
        """
        filename = file.filename.lower()

        if not any(filename.endswith(ext) for ext in self.SUPPORTED_FORMATS):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format.  Supported: {self.SUPPORTED_FORMATS}"
            )

        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum: {self.MAX_FILE_SIZE_MB}MB"
            )

        # Try parsing CSV with multiple common separators and using thousands=','
        def try_parse_csv(buf):
            candidates = [',', ';', '\t']
            best_df = None
            best_score = -1
            parse_errs = []
            for sep in candidates:
                try:
                    # use thousands=',' to handle "1,234,567" patterns
                    df_try = pd.read_csv(io.BytesIO(buf), sep=sep, thousands=',', quotechar='"', engine='python')
                    # quick score: number of numeric-like columns after coercion
                    n_numeric = 0
                    for c in df_try.columns:
                        coerced = pd.to_numeric(df_try[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
                        # count non-null numeric entries
                        n_numeric += int(coerced.notna().sum())
                    if n_numeric > best_score:
                        best_score = n_numeric
                        best_df = df_try
                except Exception as e:
                    parse_errs.append((sep, str(e)))
            return best_df, parse_errs

        try:
            if filename.endswith('.csv'):
                df, parse_errs = try_parse_csv(content)
                if df is None:
                    # fallback to default read
                    df = pd.read_csv(io.BytesIO(content), engine='python')
            elif filename.endswith('.json'):
                df = pd.read_json(io.BytesIO(content))
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise HTTPException(status_code=400, detail="Unknown file format")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

        # Normalize column names: remove BOM and strip whitespace
        df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]

        # Drop duplicated column names (keep first occurrence)
        if df.columns.duplicated().any():
            dup_names = [c for c, cnt in pd.Series(df.columns).value_counts().items() if cnt > 1]
            df = df.loc[:, ~df.columns.duplicated()]
            print(f"Dropped duplicated columns: {dup_names}")

        # Ensure preprocessor medians/clip_map loaded for heuristics
        if not preprocessor.is_loaded:
            try:
                preprocessor.load()
            except Exception:
                pass

        # Smart-fix columns that contain commas but did not parse cleanly
        df, comma_report = _smart_fix_commas(df, train_medians=preprocessor.medians)
        if comma_report:
            print("Comma normalization report:", comma_report)

        return df

    def _generate_conclusion(self, summary: Dict) -> Dict:
        """
        Generate conclusion and recommendations based on prediction results
        """
        total = summary['total_samples']
        attack_pct = summary['attack_percentage']
        benign_pct = summary['benign_percentage']
        attack_breakdown = summary['attack_breakdown']

        # Determine threat level
        if attack_pct == 0:
            threat_level = "AMAN"
            threat_color = "green"
            threat_description = "Tidak ditemukan aktivitas mencurigakan"
        elif attack_pct < 5:
            threat_level = "RENDAH"
            threat_color = "yellow"
            threat_description = "Ditemukan sedikit aktivitas mencurigakan"
        elif attack_pct < 20:
            threat_level = "SEDANG"
            threat_color = "orange"
            threat_description = "Ditemukan aktivitas mencurigakan yang perlu diperhatikan"
        elif attack_pct < 50:
            threat_level = "TINGGI"
            threat_color = "red"
            threat_description = "Ditemukan banyak aktivitas mencurigakan"
        else:
            threat_level = "KRITIS"
            threat_color = "darkred"
            threat_description = "Mayoritas traffic terdeteksi sebagai serangan"

        # Find dominant attack type
        dominant_attack = None
        dominant_attack_count = 0
        dominant_attack_pct = 0

        if attack_breakdown:
            for attack_type, data in attack_breakdown.items():
                if data['count'] > dominant_attack_count:
                    dominant_attack = attack_type
                    dominant_attack_count = data['count']
                    dominant_attack_pct = data['percentage']

        # Generate recommendations
        recommendations = []

        if attack_pct > 0:
            recommendations.append("Lakukan investigasi lebih lanjut terhadap traffic yang terdeteksi sebagai serangan")

        if "DDoS" in attack_breakdown or "DoS" in attack_breakdown:
            recommendations.append("Pertimbangkan untuk mengaktifkan rate limiting dan DDoS protection")

        if "Port Scan" in attack_breakdown:
            recommendations.append("Review firewall rules dan tutup port yang tidak diperlukan")

        if "Brute Force" in attack_breakdown:
            recommendations.append("Terapkan account lockout policy dan gunakan strong authentication")

        if attack_pct == 0:
            recommendations.append("Traffic dalam kondisi normal, tetap monitor secara berkala")

        # Build conclusion text
        conclusion_text = f"""
Dari total {total:,} flow jaringan yang dianalisis:
- {summary['benign_count']:,} flow ({benign_pct:.2f}%) terdeteksi sebagai traffic NORMAL (BENIGN)
- {summary['attack_count']:,} flow ({attack_pct:.2f}%) terdeteksi sebagai SERANGAN

Status Keamanan: {threat_level}
{threat_description}
        """.strip()

        if dominant_attack:
            conclusion_text += f"\n\nJenis serangan dominan: {dominant_attack} ({dominant_attack_pct:.2f}% dari total traffic)"

        return {
            "threat_level": threat_level,
            "threat_color": threat_color,
            "threat_description": threat_description,
            "conclusion_text": conclusion_text,
            "dominant_attack": {
                "type": dominant_attack,
                "count": dominant_attack_count,
                "percentage": dominant_attack_pct
            } if dominant_attack else None,
            "recommendations": recommendations
        }

    def _generate_detailed_summary(self, all_results: List[Dict], processing_time: float) -> Dict:
        """
        Generate detailed summary with breakdown by attack type
        """
        total_samples = len(all_results)

        # Count predictions
        prediction_counts = {}
        confidence_sum = {}
        confidence_values = {}

        for result in all_results:
            label = result['prediction']
            conf = result['confidence']

            prediction_counts[label] = prediction_counts.get(label, 0) + 1
            confidence_sum[label] = confidence_sum.get(label, 0) + conf

            if label not in confidence_values:
                confidence_values[label] = []
            confidence_values[label].append(conf)

        # Calculate statistics
        benign_count = prediction_counts.get('BENIGN', 0)
        attack_count = total_samples - benign_count

        benign_pct = (benign_count / total_samples) * 100 if total_samples > 0 else 0
        attack_pct = (attack_count / total_samples) * 100 if total_samples > 0 else 0

        # Attack breakdown
        attack_breakdown = {}
        for label, count in prediction_counts.items():
            if label != 'BENIGN':
                pct = (count / total_samples) * 100
                avg_conf = confidence_sum[label] / count if count > 0 else 0
                min_conf = min(confidence_values[label]) if confidence_values[label] else 0
                max_conf = max(confidence_values[label]) if confidence_values[label] else 0

                attack_breakdown[label] = {
                    "count": count,
                    "percentage": round(pct, 2),
                    "avg_confidence": round(avg_conf, 4),
                    "min_confidence": round(min_conf, 4),
                    "max_confidence": round(max_conf, 4)
                }

        # Sort attack breakdown by count (descending)
        attack_breakdown = dict(sorted(attack_breakdown.items(), key=lambda x: x[1]['count'], reverse=True))

        # BENIGN statistics
        benign_stats = None
        if benign_count > 0:
            avg_conf = confidence_sum.get('BENIGN', 0) / benign_count
            benign_stats = {
                "count": benign_count,
                "percentage": round(benign_pct, 2),
                "avg_confidence": round(avg_conf, 4),
                "min_confidence": round(min(confidence_values.get('BENIGN', [0])), 4),
                "max_confidence": round(max(confidence_values.get('BENIGN', [0])), 4)
            }

        summary = {
            "total_samples": total_samples,
            "processing_time_seconds": round(processing_time, 3),

            # Overview
            "benign_count": benign_count,
            "benign_percentage": round(benign_pct, 2),
            "attack_count": attack_count,
            "attack_percentage": round(attack_pct, 2),

            # Detailed breakdown
            "benign_stats": benign_stats,
            "attack_breakdown": attack_breakdown,

            # Raw counts
            "prediction_counts": prediction_counts
        }

        return summary

    def predict_dataset(
        self,
        df: pd.DataFrame,
        include_all_results: bool = False,
        batch_size: int = 1000
    ) -> Dict:
        """Predict all samples in dataset"""
        start_time = time.time()

        # Get required columns
        required_cols = preprocessor.get_feature_names()

        # Map columns using static mapping
        print("   🔄 Mapping columns...")
        df_mapped, mapping = SimpleColumnMapper.map_columns(df, required_cols)

        if mapping:
            print(f"   ✅ Mapped {len(mapping)} columns")

        # Check for missing columns
        missing = [col for col in required_cols if col not in df_mapped.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required columns",
                    "missing_columns": missing,
                    "your_columns": list(df.columns)[:20],
                    "mapping_applied": mapping,
                    "hint": "Check GET /api/v1/features for required column names"
                }
            )

        # After mapping, ensure required columns present and in order
        df_selected = df_mapped[required_cols].copy()

        # Coerce object columns that may remain (remove thousands separators, strip)
        for c in df_selected.columns:
            if df_selected[c].dtype == object:
                df_selected[c] = df_selected[c].astype(str).str.replace(',', '', regex=False).str.strip().replace({'': None})
            df_selected[c] = pd.to_numeric(df_selected[c], errors='coerce')

        # Suspicious-value check: detect values far beyond training q999 (very large)
        suspicious = []
        for c in required_cols:
            if c in preprocessor.clip_map:
                try:
                    train_q999 = preprocessor.clip_map[c][1]
                    if train_q999 is None:
                        continue
                    max_val = df_selected[c].dropna().max()
                    if pd.notna(max_val) and max_val > (float(train_q999) * 1000):
                        suspicious.append({"feature": c, "example_max": float(max_val), "train_q999": float(train_q999)})
                except Exception:
                    continue
        if suspicious:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Suspicious large values detected in uploaded file. Possible parsing/unit error.",
                    "issues": suspicious,
                    "hint": "Check delimiter/thousands separators or ensure Flow Duration units match training (seconds)."
                }
            )

        # If NaNs exist in required columns, try to fill using training medians
        if df_selected.isna().any().any():
            nan_cols = [c for c in required_cols if df_selected[c].isna().any()]
            if preprocessor._has_medians:
                print(f"Filling NaNs for columns: {nan_cols} using training medians")
                df_selected = preprocessor._fill_missing_with_medians(df_selected)
            else:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Input contains missing values and no training medians are available.",
                        "na_columns": nan_cols,
                        "hint": "Run scripts/fill_transform_medians.py to populate transform_meta.json or provide cleaned input without NaN"
                    }
                )

        total_samples = len(df_selected)
        all_results = []

        print(f"   🔮 Predicting {total_samples} samples...")

        # Process in batches
        for i in range(0, total_samples, batch_size):
            batch_df = df_selected.iloc[i:i+batch_size]
            X = preprocessor.transform(batch_df)
            # capture clip report per-batch (last run)
            clip_report = preprocessor.get_last_clip_report()
            batch_results = ids_model.predict_batch(X)
            # optionally attach clip report info into results (omitted here)
            all_results.extend(batch_results)

        processing_time = time.time() - start_time

        # Generate detailed summary
        summary = self._generate_detailed_summary(all_results, processing_time)

        # Generate conclusion
        conclusion = self._generate_conclusion(summary)

        # Store results
        self.last_dataset = df_mapped
        self.last_results = all_results
        self.last_mapping = mapping
        self.last_summary = summary

        print(f"   ✅ Complete: {total_samples} samples in {processing_time:.2f}s")
        print(f"   📊 Benign: {summary['benign_percentage']:.2f}%, Attack: {summary['attack_percentage']:.2f}%")

        response = {
            'success': True,
            'summary': summary,
            'conclusion': conclusion
        }

        if include_all_results:
            response['results'] = all_results
        else:
            response['sample_results'] = all_results[:10]

        # sanitize response to avoid NaN/inf (not JSON serializable by Starlette/JSON spec)
        response = _sanitize_for_json(response)

        return response

    def get_results_as_dataframe(self) -> Optional[pd.DataFrame]:
        """Get prediction results as DataFrame"""
        if self.last_results is None or self.last_dataset is None:
            return None

        results_df = pd.DataFrame(self.last_results)
        combined = self.last_dataset.copy()
        combined['predicted_label'] = results_df['prediction']
        combined['confidence'] = results_df['confidence']
        combined['is_attack'] = results_df['is_attack']

        return combined

    def export_results_csv(self) -> Optional[str]:
        """Export results to CSV string"""
        df = self.get_results_as_dataframe()
        if df is None:
            return None
        return df.to_csv(index=False)

    def get_summary(self) -> Optional[Dict]:
        """Get last prediction summary"""
        return self.last_summary


# Global instance
dataset_handler = DatasetHandler()