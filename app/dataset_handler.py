"""
Dataset handler for file upload and batch prediction
"""
import io
import time
import pandas as pd
from typing import Dict, List, Tuple, Optional
from fastapi import UploadFile, HTTPException

from app.preprocessing import preprocessor
from app.model import ids_model
from app.column_mapper import SimpleColumnMapper


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
        """Read uploaded file into DataFrame"""
        filename = file.filename. lower()
        
        if not any(filename.endswith(ext) for ext in self.SUPPORTED_FORMATS):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format.  Supported: {self.SUPPORTED_FORMATS}"
            )
        
        content = await file. read()
        
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum: {self.MAX_FILE_SIZE_MB}MB"
            )
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content))
            elif filename.endswith('.json'):
                df = pd.read_json(io.BytesIO(content))
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise HTTPException(status_code=400, detail="Unknown file format")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
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
            confidence_values[label]. append(conf)
        
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
            avg_conf = confidence_sum. get('BENIGN', 0) / benign_count
            benign_stats = {
                "count": benign_count,
                "percentage": round(benign_pct, 2),
                "avg_confidence": round(avg_conf, 4),
                "min_confidence": round(min(confidence_values. get('BENIGN', [0])), 4),
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
        start_time = time. time()
        
        # Get required columns
        required_cols = preprocessor.get_feature_names()
        
        # Map columns using static mapping
        print("   🔄 Mapping columns...")
        df_mapped, mapping = SimpleColumnMapper.map_columns(df, required_cols)
        
        if mapping:
            print(f"   ✅ Mapped {len(mapping)} columns")
        
        # Check for missing columns
        missing = [col for col in required_cols if col not in df_mapped. columns]
        
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
        
        total_samples = len(df_mapped)
        all_results = []
        
        print(f"   🔮 Predicting {total_samples} samples...")
        
        # Process in batches
        for i in range(0, total_samples, batch_size):
            batch_df = df_mapped.iloc[i:i+batch_size]
            X = preprocessor.transform(batch_df)
            batch_results = ids_model.predict_batch(X)
            all_results.extend(batch_results)
        
        processing_time = time.time() - start_time
        
        # Generate detailed summary
        summary = self._generate_detailed_summary(all_results, processing_time)
        
        # Generate conclusion
        conclusion = self._generate_conclusion(summary)
        
        # Store results
        self.last_dataset = df_mapped
        self. last_results = all_results
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
        return df. to_csv(index=False)
    
    def get_summary(self) -> Optional[Dict]:
        """Get last prediction summary"""
        return self.last_summary


# Global instance
dataset_handler = DatasetHandler()