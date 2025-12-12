"""
Model loading and inference
"""
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from app.config import settings


class DNNClassifier(nn.Module):
    """Deep Neural Network for intrusion detection"""
    
    def __init__(self, input_dim: int, layer_sizes: List[int], num_classes: int, 
                 activation: str = 'leaky_relu', dropout: float = 0.35):
        super().__init__()
        
        # Activation functions
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'tanh': nn.Tanh()
        }
        act_fn = activations. get(activation, nn.LeakyReLU(0.01))
        
        # Build layers as Sequential (tanpa self.network wrapper)
        layers = []
        prev_size = input_dim
        
        for hidden_size in layer_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                act_fn,
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Gunakan nn.Sequential langsung sebagai module
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class IDSModel:
    """Wrapper for IDS model with loading and inference"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.label_map = None
        self.id_to_label = None
        self.report = None
        self.device = torch.device('cuda' if torch. cuda.is_available() else 'cpu')
        self.is_loaded = False
        
    def load(self, artifacts_dir: Path = None):
        """Load model and all related artifacts"""
        if artifacts_dir is None:
            artifacts_dir = settings.ARTIFACTS_DIR
            
        # Load config
        config_path = artifacts_dir / settings.CONFIG_FILE
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Load label map
        label_map_path = artifacts_dir / settings. LABEL_MAP_FILE
        with open(label_map_path, 'r') as f:
            label_data = json.load(f)
            self.label_map = label_data['label_to_id']
            self.id_to_label = {int(k): v for k, v in label_data['id_to_label'].items()}
            
        # Load report
        report_path = artifacts_dir / settings.REPORT_FILE
        with open(report_path, 'r') as f:
            self.report = json. load(f)
        
        # Load model weights langsung sebagai Sequential
        model_path = artifacts_dir / settings.MODEL_STATE_FILE
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Cek format state_dict
        sample_key = list(state_dict.keys())[0]
        print(f"   State dict format: {sample_key}")
        
        # Build model architecture sesuai format state_dict
        if sample_key. startswith('network. ') or sample_key.startswith('model.'):
            # Format dengan wrapper
            self.model = DNNClassifier(
                input_dim=self.config['input_dim'],
                layer_sizes=self.config['layers'],
                num_classes=self.config['num_classes'],
                activation=self.config. get('activation', 'leaky_relu'),
                dropout=self.config.get('dropout', 0.35)
            )
            self.model.load_state_dict(state_dict)
        else:
            # Format Sequential langsung (seperti model Anda)
            self.model = self._build_sequential_model()
            self.model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        self.model. to(self.device)
        self.model.eval()
        
        self.is_loaded = True
        print(f"✅ Model loaded on {self.device}")
        print(f"   Architecture: {self.config['layers']}")
        print(f"   Classes: {list(self.label_map.keys())}")
    
    def _build_sequential_model(self):
        """Build Sequential model yang match dengan training"""
        layers = []
        prev_size = self.config['input_dim']
        
        activation = self.config.get('activation', 'leaky_relu')
        dropout = self.config. get('dropout', 0.35)
        
        # Activation function
        if activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.01)
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            act_fn = nn.LeakyReLU(0.01)
        
        for hidden_size in self.config['layers']:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.config['num_classes']))
        
        return nn.Sequential(*layers)
        
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on preprocessed features
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        with torch.no_grad():
            X = torch.tensor(features, dtype=torch.float32).to(self.device)
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1). cpu().numpy()
            preds = probs.argmax(axis=1)
            
        return preds, probs
    
    def predict_single(self, features: np.ndarray) -> Dict:
        """
        Predict single flow and return detailed result
        """
        preds, probs = self.predict(features)
        pred_id = int(preds[0])
        pred_label = self.id_to_label[pred_id]
        confidence = float(probs[0, pred_id])
        
        # Build probabilities dict
        prob_dict = {
            self.id_to_label[i]: float(probs[0, i]) 
            for i in range(len(self.id_to_label))
        }
        
        return {
            'prediction': pred_label,
            'confidence': confidence,
            'is_attack': pred_label != 'BENIGN',
            'probabilities': prob_dict
        }
    
    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """Predict batch of flows"""
        preds, probs = self.predict(features)
        
        results = []
        for i in range(len(preds)):
            pred_id = int(preds[i])
            pred_label = self.id_to_label[pred_id]
            confidence = float(probs[i, pred_id])
            
            prob_dict = {
                self. id_to_label[j]: float(probs[i, j]) 
                for j in range(len(self.id_to_label))
            }
            
            results.append({
                'prediction': pred_label,
                'confidence': confidence,
                'is_attack': pred_label != 'BENIGN',
                'probabilities': prob_dict
            })
            
        return results
    
    def get_info(self) -> Dict:
        """Get model information"""
        return {
            'model_name': 'DNN-IDS',
            'version': settings.APP_VERSION,
            'input_features': self.config['input_dim'],
            'num_classes': self.config['num_classes'],
            'classes': list(self.label_map.keys()),
            'architecture': {
                'layers': self.config['layers'],
                'activation': self.config.get('activation', 'leaky_relu'),
                'dropout': self.config. get('dropout', 0.35),
                'optimizer': self.config.get('optimizer', 'AdamW')
            },
            'performance': {
                'macro_f1': self.report. get('macro_f1'),
                'accuracy': self.report.get('accuracy'),
                'weighted_f1': self.report. get('weighted_f1')
            }
        }


# Global model instance
ids_model = IDSModel()