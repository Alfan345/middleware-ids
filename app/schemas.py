"""
Pydantic schemas for request/response validation
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ============== Request Schemas ==============

class PredictRequest(BaseModel):
    """Request schema for single prediction"""
    features: Dict[str, float] = Field(... , description="Dictionary of flow features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Destination Port": 80,
                    "Flow Duration": 100000,
                    "Total Fwd Packets": 10
                }
            }
        }


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction"""
    flows: List[Dict[str, float]] = Field(..., description="List of flow features")


# ============== PENTING: PredictionResult HARUS di definisi duluan ==============

class PredictionResult(BaseModel):
    """Schema for prediction result"""
    prediction: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    is_attack: bool = Field(..., description="Whether the flow is classified as an attack")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")


# ============== Basic Response Schemas ==============

class PredictResponse(BaseModel):
    """Response schema for single prediction"""
    success: bool = True
    result: PredictionResult


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction"""
    success: bool = True
    results: List[PredictionResult]
    summary: Dict[str, int] = Field(..., description="Count of each prediction class")


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    model_name: str
    version: str
    input_features: int
    num_classes: int
    classes: List[str]
    architecture: Dict
    performance: Dict


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    success: bool = False
    error: str
    detail: Optional[str] = None


# ============== Enhanced Response Schemas (untuk Dataset) ==============

class AttackTypeStats(BaseModel):
    """Statistics for each attack type"""
    count: int
    percentage: float
    avg_confidence: float
    min_confidence: float
    max_confidence: float


class BenignStats(BaseModel):
    """Statistics for benign traffic"""
    count: int
    percentage: float
    avg_confidence: float
    min_confidence: float
    max_confidence: float


class DominantAttack(BaseModel):
    """Dominant attack information"""
    type: Optional[str] = None
    count: int = 0
    percentage: float = 0.0


class Conclusion(BaseModel):
    """Conclusion and recommendations"""
    threat_level: str
    threat_color: str
    threat_description: str
    conclusion_text: str
    dominant_attack: Optional[DominantAttack] = None
    recommendations: List[str]


class EnhancedSummary(BaseModel):
    """Enhanced summary with detailed breakdown"""
    total_samples: int
    processing_time_seconds: float
    
    # Overview
    benign_count: int
    benign_percentage: float
    attack_count: int
    attack_percentage: float
    
    # Detailed stats
    benign_stats: Optional[BenignStats] = None
    attack_breakdown: Dict[str, AttackTypeStats] = Field(default_factory=dict)
    prediction_counts: Dict[str, int] = Field(default_factory=dict)


# PredictionResult sudah didefinisikan di atas, jadi ini aman
class EnhancedDatasetResponse(BaseModel):
    """Enhanced response for dataset prediction"""
    success: bool = True
    summary: EnhancedSummary
    conclusion: Conclusion
    sample_results: Optional[List[PredictionResult]] = None
    results: Optional[List[PredictionResult]] = None


# ============== Dataset Prediction Summary (Simplified) ==============

class DatasetPredictionSummary(BaseModel):
    """Summary of dataset prediction"""
    total_samples: int
    prediction_counts: Dict[str, int]
    attack_percentage: float
    benign_percentage: float
    processing_time_seconds: float


class DatasetPredictResponse(BaseModel):
    """Response schema for dataset prediction"""
    success: bool = True
    summary: DatasetPredictionSummary
    results: List[PredictionResult]


class DatasetPredictResponseLite(BaseModel):
    """Response schema for dataset prediction (without full results)"""
    success: bool = True
    summary: DatasetPredictionSummary
    sample_results: List[PredictionResult] = Field(... , description="First 10 results as sample")