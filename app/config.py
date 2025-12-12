"""
Configuration for IDS Middleware
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # App info
    APP_NAME: str = "IDS Middleware API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Deep Learning-based Intrusion Detection System Middleware"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent. parent
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    
    # Model files
    MODEL_STATE_FILE: str = "model_state.pt"
    SCALER_FILE: str = "scaler.pkl"
    TRANSFORM_META_FILE: str = "transform_meta.json"
    CONFIG_FILE: str = "config.json"
    LABEL_MAP_FILE: str = "label_map.json"
    REPORT_FILE: str = "report.json"
    
    # API Settings
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Model Settings
    CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence for attack detection
    
    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()