"""
Test cases for IDS Middleware API
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.sample_data import SAMPLE_BENIGN, SAMPLE_DDOS, SAMPLE_PORTSCAN

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and info endpoints"""
    
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True
        
    def test_model_info(self):
        response = client. get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "classes" in data
        assert len(data["classes"]) == 5
        
    def test_features_list(self):
        response = client.get("/api/v1/features")
        assert response.status_code == 200
        data = response.json()
        assert data["num_features"] == 50


class TestPrediction:
    """Test prediction endpoints"""
    
    def test_predict_benign(self):
        response = client.post(
            "/api/v1/predict",
            json={"features": SAMPLE_BENIGN}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "result" in data
        assert "prediction" in data["result"]
        assert "confidence" in data["result"]
        assert "is_attack" in data["result"]
        
    def test_predict_ddos(self):
        response = client.post(
            "/api/v1/predict",
            json={"features": SAMPLE_DDOS}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        
    def test_predict_portscan(self):
        response = client.post(
            "/api/v1/predict",
            json={"features": SAMPLE_PORTSCAN}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        
    def test_predict_missing_features(self):
        incomplete_features = {"Destination Port": 80}
        response = client.post(
            "/api/v1/predict",
            json={"features": incomplete_features}
        )
        assert response.status_code == 400
        assert "Missing features" in response.json()["detail"]


class TestBatchPrediction:
    """Test batch prediction endpoint"""
    
    def test_batch_predict(self):
        response = client. post(
            "/api/v1/predict/batch",
            json={"flows": [SAMPLE_BENIGN, SAMPLE_DDOS, SAMPLE_PORTSCAN]}
        )
        assert response.status_code == 200
        data = response. json()
        assert data["success"] == True
        assert len(data["results"]) == 3
        assert "summary" in data
        
    def test_batch_predict_empty(self):
        response = client.post(
            "/api/v1/predict/batch",
            json={"flows": []}
        )
        assert response. status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])