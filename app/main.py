"""
FastAPI application for IDS Middleware
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Dict, Optional
import time
import io

from app.config import settings
from app.model import ids_model
from app. preprocessing import preprocessor
from app.dataset_handler import dataset_handler
from app.schemas import (
    PredictRequest, PredictResponse, PredictionResult,
    BatchPredictRequest, BatchPredictResponse,
    ModelInfoResponse, HealthResponse, ErrorResponse,
    DatasetPredictResponse, DatasetPredictResponseLite, DatasetPredictionSummary
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - load models on startup"""
    print("🚀 Starting IDS Middleware...")
    
    # Load preprocessor and model
    try:
        preprocessor.load()
        ids_model.load()
        print("✅ All components loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load components: {e}")
        raise
    
    yield
    
    print("👋 Shutting down IDS Middleware...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Basic Endpoints ==============

@app. get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to IDS Middleware API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=ids_model.is_loaded,
        version=settings.APP_VERSION
    )


@app.get(f"{settings.API_PREFIX}/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information and performance metrics"""
    if not ids_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return ids_model.get_info()


@app.get(f"{settings.API_PREFIX}/features", tags=["Model"])
async def get_required_features():
    """Get list of required input features"""
    return {
        "num_features": len(preprocessor.cols),
        "features": preprocessor.get_feature_names()
    }


# ============== Single & Batch Prediction ==============

@app.post(f"{settings.API_PREFIX}/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict if a network flow is an attack or benign
    
    - **features**: Dictionary containing all 50 flow features
    """
    start_time = time.time()
    
    try:
        # Validate features
        is_valid, missing = preprocessor.validate_features(request.features)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing features: {missing}"
            )
        
        # Preprocess
        X = preprocessor. transform(request.features)
        
        # Predict
        result = ids_model.predict_single(X)
        
        return PredictResponse(
            success=True,
            result=PredictionResult(**result)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status. HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(f"{settings.API_PREFIX}/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Predict multiple network flows at once
    
    - **flows**: List of flow feature dictionaries
    """
    try:
        if not request.flows:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No flows provided"
            )
        
        # Validate all flows
        for i, flow in enumerate(request.flows):
            is_valid, missing = preprocessor.validate_features(flow)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Flow {i}: Missing features: {missing}"
                )
        
        # Preprocess batch
        X = preprocessor.transform_batch(request.flows)
        
        # Predict
        results = ids_model.predict_batch(X)
        
        # Generate summary
        summary = {}
        for r in results:
            label = r['prediction']
            summary[label] = summary.get(label, 0) + 1
        
        return BatchPredictResponse(
            success=True,
            results=[PredictionResult(**r) for r in results],
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============== Dataset Upload & Prediction ==============

@app. post(f"{settings.API_PREFIX}/predict/dataset", tags=["Dataset Prediction"])
async def predict_dataset(
    file: UploadFile = File(... , description="CSV, JSON, or Excel file with flow features"),
    include_all_results: bool = Query(False, description="Include all prediction results (may be large)")
):
    """
    Upload a dataset file and predict all samples
    
    Supported formats: CSV, JSON, Excel (. xlsx)
    
    The file must contain columns matching the required flow features.
    Use GET /api/v1/features to see the list of required features.
    
    **Note**: For large datasets, set `include_all_results=false` to get only summary and sample results.
    """
    try:
        # Read file
        df = await dataset_handler.read_file(file)
        
        print(f"📁 Uploaded file: {file.filename}")
        print(f"   Rows: {len(df)}, Columns: {len(df. columns)}")
        
        # Predict
        result = dataset_handler.predict_dataset(
            df, 
            include_all_results=include_all_results
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status. HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(f"{settings. API_PREFIX}/predict/dataset/download", tags=["Dataset Prediction"])
async def download_prediction_results():
    """
    Download the last dataset prediction results as CSV
    
    Must call POST /api/v1/predict/dataset first.
    """
    csv_content = dataset_handler.export_results_csv()
    
    if csv_content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No prediction results available.  Upload a dataset first."
        )
    
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=prediction_results. csv"
        }
    )


@app.get(f"{settings. API_PREFIX}/predict/dataset/summary", tags=["Dataset Prediction"])
async def get_last_prediction_summary():
    """
    Get summary of the last dataset prediction
    """
    if dataset_handler.last_results is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No prediction results available. Upload a dataset first."
        )
    
    results = dataset_handler.last_results
    total = len(results)
    
    prediction_counts = {}
    attack_count = 0
    
    for r in results:
        label = r['prediction']
        prediction_counts[label] = prediction_counts. get(label, 0) + 1
        if r['is_attack']:
            attack_count += 1
    
    return {
        'total_samples': total,
        'prediction_counts': prediction_counts,
        'attack_percentage': round((attack_count / total) * 100, 2),
        'benign_percentage': round(((total - attack_count) / total) * 100, 2)
    }


# ============== Run Application ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn. run(app, host="localhost", port=8000)