import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';

const BestPractices = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            ML Engineering Best Practices
          </Title>
          <Text size="xl" className="text-center mb-6">
            Building Reliable and Scalable ML Systems
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                Building production-ready ML systems requires adherence to software engineering best practices
                combined with ML-specific considerations. This includes code quality, testing strategies,
                documentation, security, and operational excellence.
              </Text>
              <List>
                <List.Item>Code organization and software architecture</List.Item>
                <List.Item>Testing strategies for ML systems</List.Item>
                <List.Item>Security and compliance considerations</List.Item>
                <List.Item>Documentation and knowledge management</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Code Organization */}
        <div data-slide className="min-h-[500px]" id="code-organization">
          <Title order={2} className="mb-6">Code Organization and Architecture</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Text size="lg">
              Well-organized code is crucial for maintainability, collaboration, and debugging in ML projects.
              A clear structure separates concerns and makes the codebase more understandable and extensible.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Project Structure</Title>
                <Code block>{`ml-project/
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .env.example
│
├── config/
│   ├── __init__.py
│   ├── development.yaml
│   ├── production.yaml
│   └── base.yaml
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py
│   │   └── selection.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── inference.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── helpers.py
│   │
│   └── api/
│       ├── __init__.py
│       ├── app.py
│       ├── routes.py
│       └── middleware.py
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── conftest.py
│
├── notebooks/
│   ├── exploration/
│   ├── experiments/
│   └── analysis/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   └── interim/
│
├── models/
│   ├── trained/
│   ├── artifacts/
│   └── registry/
│
├── docs/
│   ├── api/
│   ├── architecture/
│   └── deployment/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── deploy.py
│   └── data_pipeline.py
│
└── infrastructure/
    ├── k8s/
    ├── terraform/
    └── monitoring/`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Configuration Management</Title>
                <Code block language="python">{`# config/base.py
import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "ml_app"
    username: str = ""
    password: str = ""

@dataclass
class ModelConfig:
    name: str = "default_model"
    version: str = "1.0.0"
    path: str = "models/trained/"
    batch_size: int = 32
    max_sequence_length: int = 512
    device: str = "auto"

@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    validation_split: float = 0.2
    save_every_n_epochs: int = 5

@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    max_request_size: int = 10485760  # 10MB

@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config()
        if config_path:
            self.load_from_file(config_path)
        self.load_from_env()
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self._update_config_from_dict(config_dict)
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # Database config
        if os.getenv('DB_HOST'):
            self.config.database.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            self.config.database.port = int(os.getenv('DB_PORT'))
        
        # Model config
        if os.getenv('MODEL_PATH'):
            self.config.model.path = os.getenv('MODEL_PATH')
        if os.getenv('BATCH_SIZE'):
            self.config.model.batch_size = int(os.getenv('BATCH_SIZE'))
        
        # Environment
        self.config.environment = os.getenv('ENVIRONMENT', 'development')
        self.config.debug = os.getenv('DEBUG', 'true').lower() == 'true'
        self.config.log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update config from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self.config, section):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

# Usage
config_manager = ConfigManager(f"config/{os.getenv('ENVIRONMENT', 'development')}.yaml")
config = config_manager.config`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Testing Strategies */}
        <div data-slide className="min-h-[500px]" id="testing-strategies">
          <Title order={2} className="mb-6">Testing Strategies for ML Systems</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Unit Testing</Title>
                <Code block language="python">{`# tests/unit/test_preprocessing.py
import pytest
import numpy as np
import pandas as pd
from src.data.preprocessing import DataPreprocessor

class TestDataPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'feature1': [1, 2, 3, np.nan, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling"""
        result = preprocessor.handle_missing_values(sample_data)
        
        # Should not contain any NaN values
        assert not result.isnull().any().any()
        
        # Should maintain original shape
        assert result.shape[1] == sample_data.shape[1]
    
    def test_normalize_features(self, preprocessor):
        """Test feature normalization"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        result = preprocessor.normalize_features(data)
        
        # Check that mean is close to 0 and std is close to 1
        assert np.allclose(result.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(result.std(axis=0), 1, atol=1e-7)
    
    def test_encode_categorical_features(self, preprocessor, sample_data):
        """Test categorical encoding"""
        result = preprocessor.encode_categorical_features(
            sample_data, ['feature2']
        )
        
        # Check that categorical column is properly encoded
        encoded_cols = [col for col in result.columns if 'feature2_' in col]
        assert len(encoded_cols) > 0
        
        # Check one-hot encoding properties
        for col in encoded_cols:
            assert result[col].dtype in ['uint8', 'int64']
            assert result[col].max() <= 1
            assert result[col].min() >= 0

# tests/unit/test_model.py
import torch
import pytest
from src.models.base import BaseModel

class TestBaseModel:
    @pytest.fixture
    def model(self):
        return BaseModel(input_dim=10, hidden_dim=64, output_dim=2)
    
    def test_model_initialization(self, model):
        """Test model initializes correctly"""
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'forward')
    
    def test_forward_pass_shape(self, model):
        """Test forward pass produces correct output shape"""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 10)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_model_deterministic(self, model):
        """Test model produces deterministic output"""
        model.eval()
        input_tensor = torch.randn(1, 10)
        
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_different_batch_sizes(self, model, batch_size):
        """Test model handles different batch sizes"""
        input_tensor = torch.randn(batch_size, 10)
        output = model(input_tensor)
        
        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()

# Property-based testing
from hypothesis import given, strategies as st

class TestDataValidation:
    @given(st.integers(min_value=1, max_value=1000))
    def test_batch_size_validation(self, batch_size):
        """Property-based test for batch size validation"""
        from src.utils.validation import validate_batch_size
        
        result = validate_batch_size(batch_size)
        assert result == batch_size
    
    @given(st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=100))
    def test_feature_normalization_properties(self, features):
        """Property-based test for feature normalization"""
        from src.data.preprocessing import normalize_features
        
        if len(set(features)) > 1:  # Skip if all values are the same
            normalized = normalize_features(np.array(features).reshape(-1, 1))
            
            # Normalized features should have mean ≈ 0 and std ≈ 1
            assert abs(normalized.mean()) < 1e-10
            assert abs(normalized.std() - 1) < 1e-10`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} mb="sm">Integration and E2E Testing</Title>
                <Code block language="python">{`# tests/integration/test_training_pipeline.py
import pytest
import tempfile
import os
from src.models.training import TrainingPipeline
from src.data.ingestion import DataLoader

class TestTrainingPipeline:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_dataset(self, temp_dir):
        """Create a small sample dataset for testing"""
        import pandas as pd
        
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        data_path = os.path.join(temp_dir, 'test_data.csv')
        data.to_csv(data_path, index=False)
        return data_path
    
    def test_end_to_end_training(self, temp_dir, sample_dataset):
        """Test complete training pipeline"""
        config = {
            'data_path': sample_dataset,
            'output_dir': temp_dir,
            'epochs': 2,
            'batch_size': 16,
            'learning_rate': 0.01
        }
        
        pipeline = TrainingPipeline(config)
        
        # Run training
        model, metrics = pipeline.train()
        
        # Check that model was created
        assert model is not None
        assert 'train_loss' in metrics
        assert 'val_accuracy' in metrics
        
        # Check that model files were saved
        model_files = os.listdir(temp_dir)
        assert any('model' in f for f in model_files)
    
    def test_data_pipeline_integration(self, temp_dir, sample_dataset):
        """Test data loading and preprocessing integration"""
        loader = DataLoader(sample_dataset)
        data = loader.load()
        
        # Check data loading
        assert data is not None
        assert len(data) > 0
        
        # Test preprocessing integration
        from src.data.preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(data)
        
        # Check preprocessing results
        assert processed_data.shape[0] == data.shape[0]
        assert not processed_data.isnull().any().any()

# tests/e2e/test_api.py
import pytest
import requests
import docker
import time
from contextlib import contextmanager

@contextmanager
def docker_service():
    """Context manager to run Docker service for testing"""
    client = docker.from_env()
    
    # Build and start container
    container = client.containers.run(
        "ml-model-api:test",
        ports={'8000/tcp': 8001},
        detach=True,
        remove=True
    )
    
    # Wait for service to be ready
    max_retries = 30
    for _ in range(max_retries):
        try:
            response = requests.get("http://localhost:8001/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(1)
    
    try:
        yield "http://localhost:8001"
    finally:
        container.stop()

class TestAPIEndToEnd:
    def test_health_endpoint(self):
        """Test API health endpoint"""
        with docker_service() as base_url:
            response = requests.get(f"{base_url}/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint"""
        with docker_service() as base_url:
            payload = {
                "features": [[1.0, 2.0, 3.0]],
                "model_version": "latest"
            }
            
            response = requests.post(
                f"{base_url}/predict", 
                json=payload
            )
            
            assert response.status_code == 200
            
            result = response.json()
            assert "predictions" in result
            assert "model_version" in result
            assert len(result["predictions"]) == 1
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        with docker_service() as base_url:
            payload = {
                "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "model_version": "latest"
            }
            
            response = requests.post(
                f"{base_url}/predict", 
                json=payload
            )
            
            assert response.status_code == 200
            
            result = response.json()
            assert len(result["predictions"]) == 2

# Performance testing
import pytest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class TestPerformance:
    def test_prediction_latency(self):
        """Test that predictions meet latency requirements"""
        from src.models.inference import ModelInference
        
        model = ModelInference("models/test_model.pt")
        input_data = torch.randn(1, 10)
        
        # Warmup
        for _ in range(10):
            model.predict(input_data)
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.time()
            model.predict(input_data)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[95]
        
        # Assert performance requirements
        assert avg_latency < 50  # 50ms average
        assert p95_latency < 100  # 100ms P95
    
    def test_concurrent_predictions(self):
        """Test model performance under concurrent load"""
        from src.models.inference import ModelInference
        
        model = ModelInference("models/test_model.pt")
        
        def make_prediction():
            input_data = torch.randn(1, 10)
            return model.predict(input_data)
        
        # Test with multiple concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction) for _ in range(50)]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # All predictions should succeed
        assert len(results) == 50
        assert all(r is not None for r in results)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: Security and Compliance */}
        <div data-slide className="min-h-[500px]" id="security-compliance">
          <Title order={2} className="mb-6">Security and Compliance</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">Security Best Practices</Title>
                <Code block language="python">{`# Security configuration
import secrets
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecurityManager:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self):
        """Get encryption key from environment or create new one"""
        import os
        
        key_env = os.getenv('ENCRYPTION_KEY')
        if key_env:
            return key_env.encode()
        
        # Generate new key (in production, store this securely)
        password = os.getenv('MASTER_PASSWORD', 'default-password').encode()
        salt = os.getenv('SALT', 'default-salt').encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def hash_pii(self, pii_data: str) -> str:
        """Hash PII data for logging/analytics"""
        return hashlib.sha256(pii_data.encode()).hexdigest()

# Input validation and sanitization
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import re

class PredictionRequest(BaseModel):
    features: List[List[float]] = Field(..., min_items=1, max_items=1000)
    model_version: Optional[str] = Field("latest", regex=r"^[a-zA-Z0-9._-]+$")
    user_id: Optional[str] = Field(None, max_length=100)
    
    @validator('features')
    def validate_features(cls, v):
        for feature_vector in v:
            if len(feature_vector) > 1000:  # Limit feature vector size
                raise ValueError("Feature vector too large")
            
            for feature in feature_vector:
                if not isinstance(feature, (int, float)):
                    raise ValueError("Features must be numeric")
                if abs(feature) > 1e6:  # Prevent extremely large values
                    raise ValueError("Feature values too large")
        
        return v
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid user ID format")
        return v

# Authentication and authorization
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app

class AuthManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def generate_token(self, user_id: str, permissions: List[str], 
                      expires_in_hours: int = 24) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

def require_auth(required_permission: str = None):
    """Decorator for API authentication"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'No token provided'}), 401
            
            try:
                # Remove 'Bearer ' prefix
                token = token.replace('Bearer ', '')
                
                auth_manager = current_app.config['auth_manager']
                payload = auth_manager.verify_token(token)
                
                # Check permissions
                if required_permission:
                    user_permissions = payload.get('permissions', [])
                    if required_permission not in user_permissions:
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                # Add user info to request context
                request.user_id = payload['user_id']
                request.permissions = payload['permissions']
                
                return f(*args, **kwargs)
            
            except ValueError as e:
                return jsonify({'error': str(e)}), 401
        
        return decorated_function
    return decorator

# Audit logging
class AuditLogger:
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.logger = logging.getLogger('audit')
    
    def log_api_access(self, user_id: str, endpoint: str, 
                      request_data: dict, response_status: int):
        """Log API access for audit purposes"""
        # Hash sensitive data
        sanitized_data = self._sanitize_request_data(request_data)
        
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'api_access',
            'user_id_hash': self.security_manager.hash_pii(user_id),
            'endpoint': endpoint,
            'request_data': sanitized_data,
            'response_status': response_status,
            'ip_address': self._get_client_ip()
        }
        
        self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    def log_model_access(self, user_id: str, model_version: str, 
                        input_shape: tuple, prediction_confidence: float):
        """Log model access and predictions"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'model_prediction',
            'user_id_hash': self.security_manager.hash_pii(user_id),
            'model_version': model_version,
            'input_shape': input_shape,
            'prediction_confidence': prediction_confidence
        }
        
        self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    def _sanitize_request_data(self, data: dict) -> dict:
        """Remove or hash sensitive data from request"""
        sanitized = data.copy()
        
        # Remove sensitive fields
        sensitive_fields = ['password', 'token', 'api_key', 'ssn', 'credit_card']
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = '[REDACTED]'
        
        return sanitized`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} mb="sm">Data Privacy and Compliance</Title>
                <Code block language="python">{`# GDPR compliance utilities
from typing import List, Dict, Any
import json
from datetime import datetime, timedelta

class PrivacyManager:
    def __init__(self, data_retention_days: int = 365):
        self.data_retention_days = data_retention_days
        self.consent_storage = {}  # In production, use a database
    
    def record_consent(self, user_id: str, consent_type: str, 
                      data_purposes: List[str], consent_given: bool):
        """Record user consent for GDPR compliance"""
        consent_record = {
            'user_id': user_id,
            'consent_type': consent_type,
            'data_purposes': data_purposes,
            'consent_given': consent_given,
            'timestamp': datetime.utcnow().isoformat(),
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent()
        }
        
        self.consent_storage[f"{user_id}:{consent_type}"] = consent_record
        return consent_record
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for specific data purpose"""
        for key, record in self.consent_storage.items():
            if (record['user_id'] == user_id and 
                purpose in record['data_purposes'] and
                record['consent_given']):
                return True
        return False
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get all data associated with a user (for GDPR data portability)"""
        user_data = {
            'user_id': user_id,
            'consent_records': [],
            'predictions': [],
            'data_processed': []
        }
        
        # Collect consent records
        for record in self.consent_storage.values():
            if record['user_id'] == user_id:
                user_data['consent_records'].append(record)
        
        # In production, collect data from all relevant systems
        return user_data
    
    def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete all data associated with a user (GDPR right to be forgotten)"""
        deletion_log = {
            'user_id': user_id,
            'deletion_timestamp': datetime.utcnow().isoformat(),
            'deleted_records': []
        }
        
        # Delete consent records
        keys_to_delete = [
            key for key, record in self.consent_storage.items()
            if record['user_id'] == user_id
        ]
        
        for key in keys_to_delete:
            del self.consent_storage[key]
            deletion_log['deleted_records'].append(f"consent:{key}")
        
        # In production, delete from all relevant systems
        return deletion_log
    
    def anonymize_data(self, data: Dict[str, Any], 
                      anonymization_fields: List[str]) -> Dict[str, Any]:
        """Anonymize data by removing or hashing PII"""
        anonymized = data.copy()
        
        for field in anonymization_fields:
            if field in anonymized:
                if isinstance(anonymized[field], str):
                    # Hash the field
                    anonymized[field] = hashlib.sha256(
                        anonymized[field].encode()
                    ).hexdigest()
                else:
                    # Remove the field
                    del anonymized[field]
        
        return anonymized

# Model bias and fairness monitoring
class FairnessMonitor:
    def __init__(self):
        self.protected_attributes = ['gender', 'race', 'age_group']
        self.fairness_metrics = {}
    
    def evaluate_bias(self, predictions: List[float], 
                     protected_attributes: Dict[str, List[Any]], 
                     ground_truth: List[int] = None) -> Dict[str, float]:
        """Evaluate model bias across protected groups"""
        bias_results = {}
        
        for attr_name, attr_values in protected_attributes.items():
            # Group predictions by attribute value
            groups = {}
            for i, attr_val in enumerate(attr_values):
                if attr_val not in groups:
                    groups[attr_val] = {'predictions': [], 'ground_truth': []}
                
                groups[attr_val]['predictions'].append(predictions[i])
                if ground_truth:
                    groups[attr_val]['ground_truth'].append(ground_truth[i])
            
            # Calculate fairness metrics
            group_metrics = {}
            for group_name, group_data in groups.items():
                preds = group_data['predictions']
                group_metrics[group_name] = {
                    'mean_prediction': np.mean(preds),
                    'prediction_rate': np.mean([p > 0.5 for p in preds])
                }
                
                if ground_truth:
                    gt = group_data['ground_truth']
                    binary_preds = [1 if p > 0.5 else 0 for p in preds]
                    group_metrics[group_name]['accuracy'] = np.mean(
                        [p == g for p, g in zip(binary_preds, gt)]
                    )
            
            # Calculate disparate impact
            prediction_rates = [
                metrics['prediction_rate'] 
                for metrics in group_metrics.values()
            ]
            
            disparate_impact = min(prediction_rates) / max(prediction_rates)
            
            bias_results[attr_name] = {
                'group_metrics': group_metrics,
                'disparate_impact': disparate_impact,
                'bias_detected': disparate_impact < 0.8  # 80% rule
            }
        
        return bias_results
    
    def log_fairness_metrics(self, bias_results: Dict[str, Any]):
        """Log fairness metrics for monitoring"""
        timestamp = datetime.utcnow().isoformat()
        
        for attr_name, results in bias_results.items():
            if results['bias_detected']:
                logger.warning(
                    f"Potential bias detected for {attr_name}: "
                    f"Disparate impact = {results['disparate_impact']:.3f}"
                )
            
            # Store metrics for trend analysis
            if attr_name not in self.fairness_metrics:
                self.fairness_metrics[attr_name] = []
            
            self.fairness_metrics[attr_name].append({
                'timestamp': timestamp,
                'disparate_impact': results['disparate_impact'],
                'bias_detected': results['bias_detected']
            })

# Secure model serving
class SecureModelServer:
    def __init__(self, model_path: str, security_manager: SecurityManager):
        self.model = self._load_model_securely(model_path)
        self.security_manager = security_manager
        self.rate_limiter = self._setup_rate_limiting()
    
    def _load_model_securely(self, model_path: str):
        """Load model with security checks"""
        # Verify model file integrity
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check file permissions
        file_stat = os.stat(model_path)
        if file_stat.st_mode & 0o077:  # Check for world/group permissions
            logger.warning(f"Model file {model_path} has overly permissive permissions")
        
        # Load model in restricted environment
        try:
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model securely: {e}")
            raise
    
    def predict_securely(self, features: List[List[float]], 
                        user_id: str) -> Dict[str, Any]:
        """Make predictions with security checks"""
        # Rate limiting
        if not self.rate_limiter.allow_request(user_id):
            raise ValueError("Rate limit exceeded")
        
        # Input validation
        validated_features = self._validate_input(features)
        
        # Make prediction in secure context
        with torch.no_grad():
            input_tensor = torch.tensor(validated_features, dtype=torch.float32)
            predictions = self.model(input_tensor)
            
            # Apply output sanitization
            sanitized_predictions = self._sanitize_output(predictions)
            
            return {
                'predictions': sanitized_predictions.tolist(),
                'model_version': 'secure_v1.0',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _validate_input(self, features: List[List[float]]) -> torch.Tensor:
        """Validate and sanitize input features"""
        if len(features) > 100:  # Limit batch size
            raise ValueError("Batch size too large")
        
        for feature_vector in features:
            if len(feature_vector) != 10:  # Expected feature count
                raise ValueError("Invalid feature vector length")
            
            for feature in feature_vector:
                if not (-100 <= feature <= 100):  # Range check
                    raise ValueError("Feature value out of acceptable range")
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _sanitize_output(self, predictions: torch.Tensor) -> torch.Tensor:
        """Sanitize model output"""
        # Apply softmax for probability interpretation
        probabilities = torch.softmax(predictions, dim=1)
        
        # Ensure outputs are in valid range
        probabilities = torch.clamp(probabilities, 0.0, 1.0)
        
        return probabilities`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 5: Documentation and Knowledge Management */}
        <div data-slide className="min-h-[500px]" id="documentation">
          <Title order={2} className="mb-6">Documentation and Knowledge Management</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-indigo-50">
                <Title order={4} mb="sm">Code Documentation</Title>
                <Code block language="python">{`"""
Machine Learning Model Training Module

This module provides comprehensive functionality for training, validating,
and persisting machine learning models with enterprise-grade features
including monitoring, versioning, and reproducibility.

Example:
    >>> from src.models.training import ModelTrainer
    >>> config = TrainingConfig(epochs=50, learning_rate=0.001)
    >>> trainer = ModelTrainer(config)
    >>> model, metrics = trainer.train(train_data, val_data)
    >>> trainer.save_model(model, "model_v1.0.pt")

Author: ML Engineering Team
Version: 2.1.0
Last Updated: 2024-01-15
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration class for model training parameters.
    
    This class encapsulates all hyperparameters and settings required
    for training machine learning models, ensuring reproducibility
    and easy parameter management.
    
    Attributes:
        epochs (int): Number of training epochs. Must be positive.
        learning_rate (float): Initial learning rate for optimizer.
            Typical values: 0.001-0.1 for SGD, 0.0001-0.01 for Adam.
        batch_size (int): Mini-batch size for training. Powers of 2 
            (16, 32, 64, 128) often work best for GPU optimization.
        weight_decay (float): L2 regularization coefficient. 
            Range: 0.0-0.01. Higher values increase regularization.
        scheduler_type (str): Learning rate scheduler type.
            Options: 'cosine', 'step', 'exponential', 'reduce_on_plateau'
        validation_split (float): Fraction of data for validation.
            Range: 0.1-0.3. Default 0.2 provides good balance.
        early_stopping_patience (int): Epochs to wait before stopping
            when validation loss stops improving.
        model_checkpoint_every (int): Save model checkpoint every N epochs.
        
    Raises:
        ValueError: If any parameter is outside valid range.
        
    Example:
        >>> config = TrainingConfig(
        ...     epochs=100,
        ...     learning_rate=0.001,
        ...     batch_size=32,
        ...     weight_decay=0.01
        ... )
    """
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 32
    weight_decay: float = 0.01
    scheduler_type: str = 'cosine'
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    model_checkpoint_every: int = 5
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError(f"learning_rate must be in (0,1), got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

class ModelTrainer:
    """Production-grade model trainer with monitoring and reproducibility.
    
    This class provides a comprehensive training framework that includes:
    - Automatic mixed precision training for performance
    - Learning rate scheduling and early stopping
    - Model checkpointing and versioning
    - Training metrics logging and monitoring
    - Gradient clipping and stability checks
    - Reproducible training with seed management
    
    The trainer is designed for enterprise ML workflows where reliability,
    monitoring, and reproducibility are critical requirements.
    
    Attributes:
        config (TrainingConfig): Training configuration parameters
        model (nn.Module): PyTorch model to train
        optimizer (torch.optim.Optimizer): Optimizer instance
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler
        scaler (torch.cuda.amp.GradScaler): Mixed precision scaler
        
    Example:
        >>> config = TrainingConfig(epochs=50, learning_rate=0.001)
        >>> trainer = ModelTrainer(config)
        >>> model, metrics = trainer.train(train_loader, val_loader)
        >>> print(f"Final accuracy: {metrics['val_accuracy']:.3f}")
    """
    
    def __init__(self, config: TrainingConfig, device: str = 'auto'):
        """Initialize the model trainer.
        
        Args:
            config: Training configuration with hyperparameters
            device: Training device ('cpu', 'cuda', or 'auto' for automatic)
            
        Raises:
            RuntimeError: If CUDA requested but not available
            ValueError: If configuration is invalid
        """
        self.config = config
        self.device = self._setup_device(device)
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        logger.info(f"ModelTrainer initialized with device: {self.device}")
        logger.info(f"Training configuration: {self.config}")
    
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Train the model with comprehensive monitoring and validation.
        
        This method implements a complete training loop with:
        - Automatic mixed precision for GPU acceleration
        - Learning rate scheduling based on validation performance
        - Early stopping to prevent overfitting
        - Regular model checkpointing for recovery
        - Comprehensive metrics logging for analysis
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data  
            model: PyTorch model to train
            
        Returns:
            Tuple containing:
                - Trained model (nn.Module)
                - Training metrics dictionary with keys:
                    * 'train_loss': List of training losses per epoch
                    * 'val_loss': List of validation losses per epoch
                    * 'val_accuracy': List of validation accuracies per epoch
                    * 'best_val_loss': Best validation loss achieved
                    * 'best_epoch': Epoch with best validation performance
                    * 'total_train_time': Total training time in seconds
                    
        Raises:
            RuntimeError: If training fails due to CUDA OOM or model issues
            ValueError: If data loaders are empty or incompatible
            
        Example:
            >>> trainer = ModelTrainer(config)
            >>> model = MyModel(input_dim=784, num_classes=10)
            >>> trained_model, metrics = trainer.train(train_dl, val_dl, model)
            >>> logger.info(f"Training completed. Best accuracy: {metrics['best_val_accuracy']:.3f}")
        """
        
        # Implementation would go here...
        # This is just showing the documentation structure
        
        pass
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup training device with automatic detection.
        
        Args:
            device: Device specification ('cpu', 'cuda', 'auto')
            
        Returns:
            torch.device: Configured device for training
            
        Raises:
            RuntimeError: If CUDA requested but not available
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        
        return torch.device(device)

# Model cards for documentation
MODEL_CARD_TEMPLATE = """
# Model Card: {model_name}

## Model Overview
- **Model Name**: {model_name}
- **Model Version**: {model_version}
- **Model Type**: {model_type}
- **Framework**: PyTorch {pytorch_version}
- **Created**: {creation_date}
- **Last Updated**: {update_date}

## Intended Use
### Primary Use Cases
{primary_use_cases}

### Out-of-Scope Use Cases  
{out_of_scope_use_cases}

### Target Users
{target_users}

## Model Architecture
- **Input Shape**: {input_shape}
- **Output Shape**: {output_shape}
- **Number of Parameters**: {num_parameters:,}
- **Model Size**: {model_size_mb:.1f} MB

## Training Data
- **Dataset**: {dataset_name}
- **Training Samples**: {train_samples:,}
- **Validation Samples**: {val_samples:,}
- **Features**: {num_features}
- **Data Collection Period**: {data_period}

## Performance Metrics
### Overall Performance
- **Accuracy**: {accuracy:.3f} ± {accuracy_std:.3f}
- **Precision**: {precision:.3f} ± {precision_std:.3f}
- **Recall**: {recall:.3f} ± {recall_std:.3f}
- **F1-Score**: {f1_score:.3f} ± {f1_std:.3f}

### Performance by Subgroup
{subgroup_performance}

## Limitations and Biases
{limitations_and_biases}

## Ethical Considerations
{ethical_considerations}

## Technical Specifications
- **Training Time**: {training_time}
- **Inference Time**: {inference_time_ms:.1f}ms (avg)
- **Memory Usage**: {memory_usage_mb:.1f}MB
- **Dependencies**: {dependencies}

## Model Governance
- **Model Owner**: {model_owner}
- **Review Process**: {review_process}
- **Update Frequency**: {update_frequency}
- **Monitoring**: {monitoring_description}
"""

def generate_model_card(model: nn.Module, 
                       training_data: Dict[str, Any],
                       metrics: Dict[str, float],
                       metadata: Dict[str, Any]) -> str:
    """Generate comprehensive model documentation card.
    
    Args:
        model: Trained PyTorch model
        training_data: Information about training dataset
        metrics: Model performance metrics
        metadata: Additional model metadata
        
    Returns:
        str: Formatted model card in Markdown format
    """
    return MODEL_CARD_TEMPLATE.format(**metadata)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">API Documentation</Title>
                <Code block language="python">{`# API documentation with OpenAPI/Swagger
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, marshal_with
from flask_restx import reqparse

app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='ML Model API',
    description='Production ML model serving API with comprehensive documentation',
    doc='/docs/',  # Swagger UI endpoint
    contact='ml-team@company.com',
    license='MIT'
)

# Define API models for documentation
prediction_request = api.model('PredictionRequest', {
    'features': fields.List(
        fields.List(fields.Float),
        required=True,
        description='List of feature vectors for batch prediction',
        example=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    ),
    'model_version': fields.String(
        required=False,
        description='Specific model version to use',
        default='latest',
        example='v2.1.0'
    ),
    'return_probabilities': fields.Boolean(
        required=False,
        description='Whether to return class probabilities',
        default=False
    )
})

prediction_response = api.model('PredictionResponse', {
    'predictions': fields.List(
        fields.Float,
        description='Model predictions for input features',
        example=[0.85, 0.23]
    ),
    'probabilities': fields.List(
        fields.List(fields.Float),
        description='Class probabilities (if requested)',
        example=[[0.15, 0.85], [0.77, 0.23]]
    ),
    'model_version': fields.String(
        description='Version of model used for prediction',
        example='v2.1.0'
    ),
    'inference_time_ms': fields.Float(
        description='Time taken for inference in milliseconds',
        example=23.5
    ),
    'timestamp': fields.String(
        description='ISO timestamp of prediction',
        example='2024-01-15T10:30:45Z'
    )
})

health_response = api.model('HealthResponse', {
    'status': fields.String(
        description='Service health status',
        enum=['healthy', 'degraded', 'unhealthy'],
        example='healthy'
    ),
    'model_loaded': fields.Boolean(
        description='Whether model is loaded and ready',
        example=True
    ),
    'version': fields.String(
        description='API version',
        example='1.0.0'
    ),
    'uptime_seconds': fields.Integer(
        description='Service uptime in seconds',
        example=86400
    )
})

@api.route('/predict')
class Predict(Resource):
    @api.doc(
        description='''
        Make predictions using the trained ML model.
        
        This endpoint accepts a batch of feature vectors and returns predictions.
        Features should be normalized using the same preprocessing pipeline
        used during model training.
        
        **Rate Limits**: 100 requests per minute per API key
        **Max Batch Size**: 1000 samples per request
        **Timeout**: 30 seconds
        
        **Error Codes**:
        - 400: Invalid input data or format
        - 401: Authentication required
        - 429: Rate limit exceeded
        - 500: Internal server error
        ''',
        responses={
            200: 'Successful prediction',
            400: 'Bad Request - Invalid input',
            401: 'Unauthorized - Invalid API key',
            429: 'Too Many Requests - Rate limit exceeded',
            500: 'Internal Server Error'
        }
    )
    @api.expect(prediction_request, validate=True)
    @api.marshal_with(prediction_response)
    def post(self):
        """Make batch predictions on input features"""
        try:
            data = request.get_json()
            
            # Validate input
            if 'features' not in data:
                api.abort(400, 'Missing required field: features')
            
            features = data['features']
            if len(features) > 1000:
                api.abort(400, 'Batch size exceeds maximum of 1000')
            
            # Make predictions (implementation details omitted)
            predictions = model_predictor.predict(features)
            
            response = {
                'predictions': predictions.tolist(),
                'model_version': model_predictor.version,
                'inference_time_ms': 25.3,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            return response
            
        except Exception as e:
            api.abort(500, f'Prediction failed: {str(e)}')

@api.route('/health')
class Health(Resource):
    @api.doc(
        description='Check service health and readiness',
        responses={
            200: 'Service is healthy',
            503: 'Service is unhealthy'
        }
    )
    @api.marshal_with(health_response)
    def get(self):
        """Health check endpoint for monitoring and load balancers"""
        try:
            # Check model availability
            model_loaded = hasattr(model_predictor, 'model') and model_predictor.model is not None
            
            # Check system resources
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Determine health status
            if cpu_percent > 90 or memory_percent > 95 or not model_loaded:
                status = 'unhealthy'
                status_code = 503
            elif cpu_percent > 70 or memory_percent > 80:
                status = 'degraded'
                status_code = 200
            else:
                status = 'healthy'
                status_code = 200
            
            response = {
                'status': status,
                'model_loaded': model_loaded,
                'version': '1.0.0',
                'uptime_seconds': int(time.time() - start_time)
            }
            
            return response, status_code
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': False,
                'version': '1.0.0',
                'uptime_seconds': 0
            }, 503

# Architecture Decision Records (ADRs)
ADR_TEMPLATE = """
# ADR-{number}: {title}

## Status
{status}

## Context
{context}

## Decision
{decision}

## Consequences
{consequences}

## Alternatives Considered
{alternatives}

---
- **Date**: {date}
- **Authors**: {authors}
- **Reviewers**: {reviewers}
"""

# Runbook template
RUNBOOK_TEMPLATE = """
# ML System Runbook

## System Overview
- **Service Name**: {service_name}
- **Purpose**: {purpose}
- **Dependencies**: {dependencies}
- **SLA**: {sla}

## Architecture
{architecture_description}

## Monitoring and Alerts
### Key Metrics
- Request latency (P95 < 100ms)
- Error rate (< 1%)
- Model accuracy (> 95%)
- Data drift score (< 0.2)

### Alert Definitions
{alert_definitions}

## Common Issues and Solutions

### High Latency
**Symptoms**: P95 latency > 100ms
**Causes**:
- Model inference bottleneck
- Database connection issues
- Resource contention

**Investigation**:
1. Check model server metrics
2. Verify database performance
3. Review system resources

**Resolution**:
- Scale model replicas
- Optimize model inference
- Add connection pooling

### Data Drift Detected
**Symptoms**: Drift monitoring alerts
**Causes**:
- Changes in data source
- Seasonal variations
- Data quality issues

**Investigation**:
1. Review drift metrics by feature
2. Compare recent vs historical data
3. Check data pipeline health

**Resolution**:
- Trigger model retraining
- Update feature preprocessing
- Investigate data source changes

## Deployment Procedures
{deployment_procedures}

## Recovery Procedures
{recovery_procedures}

## Contact Information
- **On-call**: {oncall_contact}
- **Team Lead**: {team_lead}
- **Escalation**: {escalation_contact}
"""
`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 6: Summary and Checklist */}
        <div data-slide className="min-h-[500px]" id="summary-checklist">
          <Title order={2} className="mb-6">ML Engineering Checklist</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Development Phase</Title>
                <List spacing="xs" size="sm">
                  <List.Item>✅ Code follows consistent style guidelines (PEP 8, Black)</List.Item>
                  <List.Item>✅ Project structure follows established conventions</List.Item>
                  <List.Item>✅ Configuration management implemented</List.Item>
                  <List.Item>✅ Environment isolation with containers/virtual envs</List.Item>
                  <List.Item>✅ Version control with meaningful commit messages</List.Item>
                  <List.Item>✅ Dependency management with pinned versions</List.Item>
                  <List.Item>✅ Unit tests cover critical functionality</List.Item>
                  <List.Item>✅ Integration tests validate component interactions</List.Item>
                  <List.Item>✅ Data validation and schema enforcement</List.Item>
                  <List.Item>✅ Model validation tests implemented</List.Item>
                  <List.Item>✅ Performance benchmarks established</List.Item>
                  <List.Item>✅ Code documentation with docstrings</List.Item>
                </List>
                
                <Paper className="p-3 bg-white mt-4">
                  <Title order={5} className="mb-2">Code Quality Tools</Title>
                  <List size="sm">
                    <List.Item>🔧 Black/autopep8 for formatting</List.Item>
                    <List.Item>🔍 Flake8/pylint for linting</List.Item>
                    <List.Item>📝 mypy for type checking</List.Item>
                    <List.Item>🧪 pytest for testing</List.Item>
                    <List.Item>📊 coverage for test coverage</List.Item>
                    <List.Item>🔒 bandit for security scanning</List.Item>
                  </List>
                </Paper>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Production Phase</Title>
                <List spacing="xs" size="sm">
                  <List.Item>✅ Security scanning and vulnerability assessment</List.Item>
                  <List.Item>✅ Authentication and authorization implemented</List.Item>
                  <List.Item>✅ Input validation and sanitization</List.Item>
                  <List.Item>✅ Rate limiting and DDoS protection</List.Item>
                  <List.Item>✅ Comprehensive logging and audit trails</List.Item>
                  <List.Item>✅ Monitoring and alerting configured</List.Item>
                  <List.Item>✅ Health checks and readiness probes</List.Item>
                  <List.Item>✅ Data drift detection implemented</List.Item>
                  <List.Item>✅ Model performance monitoring</List.Item>
                  <List.Item>✅ Automated backup and recovery procedures</List.Item>
                  <List.Item>✅ CI/CD pipeline with automated testing</List.Item>
                  <List.Item>✅ Documentation updated and accessible</List.Item>
                </List>
                
                <Paper className="p-3 bg-white mt-4">
                  <Title order={5} className="mb-2">Compliance Requirements</Title>
                  <List size="sm">
                    <List.Item>📋 Data privacy compliance (GDPR/CCPA)</List.Item>
                    <List.Item>🔐 Security standards adherence</List.Item>
                    <List.Item>📊 Model bias and fairness evaluation</List.Item>
                    <List.Item>📝 Model cards and documentation</List.Item>
                    <List.Item>🔍 Audit trail maintenance</List.Item>
                    <List.Item>⚖️ Regulatory compliance verification</List.Item>
                  </List>
                </Paper>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-6 bg-yellow-50 mt-6">
            <Title order={4} className="mb-4 text-center">Key Success Factors</Title>
            <Grid gutter="lg">
              <Grid.Col span={4}>
                <div className="text-center">
                  <Title order={5} className="mb-2">🏗️ Architecture</Title>
                  <Text size="sm">
                    Modular, testable, and maintainable code structure with clear separation of concerns
                  </Text>
                </div>
              </Grid.Col>
              <Grid.Col span={4}>
                <div className="text-center">
                  <Title order={5} className="mb-2">🛡️ Reliability</Title>
                  <Text size="sm">
                    Robust error handling, monitoring, and automated recovery mechanisms
                  </Text>
                </div>
              </Grid.Col>
              <Grid.Col span={4}>
                <div className="text-center">
                  <Title order={5} className="mb-2">📖 Documentation</Title>
                  <Text size="sm">
                    Comprehensive, up-to-date documentation for development and operations
                  </Text>
                </div>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default BestPractices;