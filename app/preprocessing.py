"""
Preprocessing pipeline for flow features
Matches the transform pipeline used during training
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
        self. medians = None
        self.is_loaded = False
        
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
        self.heavy_cols = meta['heavy_cols']
        self.medians = meta. get('medians', {})
        
        self.is_loaded = True
        print(f"✅ Preprocessor loaded: {len(self.cols)} features")
        
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
            df = features. copy()
            
        # Select and order columns
        df = df[self.cols]. copy()
        
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Fill missing values with medians
        for c in self.cols:
            if df[c].isna().any():
                fill_val = self.medians. get(c, df[c]. median())
                df[c] = df[c].fillna(fill_val)
        
        # Apply log transform to heavy-tailed columns
        for c in self.heavy_cols:
            if c in df.columns:
                df[c] = np.log1p(np.clip(df[c], 0, None))
        
        # Scale features
        X_scaled = self.scaler. transform(df. values). astype('float32')
        
        return X_scaled
    
    def transform_batch(self, flows: List[Dict[str, float]]) -> np.ndarray:
        """Transform a batch of flow features"""
        df = pd.DataFrame(flows)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get list of required feature names"""
        return self.cols. copy()
    
    def validate_features(self, features: Dict[str, float]) -> tuple:
        """
        Validate that all required features are present
        
        Returns:
            (is_valid, missing_features)
        """
        missing = [col for col in self.cols if col not in features]
        return len(missing) == 0, missing


# Global preprocessor instance
preprocessor = FlowPreprocessor()