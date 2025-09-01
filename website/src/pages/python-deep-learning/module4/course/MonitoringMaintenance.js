import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';

const MonitoringMaintenance = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Monitoring and Maintenance
          </Title>
          <Text size="xl" className="text-center mb-6">
            Keeping ML Models Healthy in Production
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                Production ML systems require continuous monitoring to ensure optimal performance,
                detect data drift, identify model degradation, and maintain system reliability.
                Effective monitoring enables proactive maintenance and quick issue resolution.
              </Text>
              <List>
                <List.Item>Performance monitoring and alerting systems</List.Item>
                <List.Item>Data drift detection and model degradation</List.Item>
                <List.Item>Logging, metrics, and observability</List.Item>
                <List.Item>Automated retraining and model updates</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Performance Monitoring */}
        <div data-slide className="min-h-[500px]" id="performance-monitoring">
          <Title order={2} mb="xl">Performance Monitoring</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Text size="lg">
              Performance monitoring tracks key metrics like latency, throughput, accuracy, and resource usage
              to ensure the model meets service level objectives and provides early warning of issues.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Metrics Collection</Title>
                <Code block language="python">{`import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, Any
import logging

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('ml_request_duration_seconds', 'Request latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy')
GPU_MEMORY_USAGE = Gauge('ml_gpu_memory_bytes', 'GPU memory usage')
CPU_USAGE = Gauge('ml_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('ml_memory_usage_bytes', 'Memory usage in bytes')

class ModelMonitor:
    def __init__(self, model, model_name: str = "default"):
        self.model = model
        self.model_name = model_name
        self.request_times = []
        self.predictions = []
        self.ground_truth = []
        
        # Start metrics server
        start_http_server(8001)
        
    def log_request(self, input_data, prediction, latency_ms, status="success"):
        """Log request metrics"""
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=status).inc()
        REQUEST_LATENCY.observe(latency_ms / 1000.0)
        
        self.request_times.append(time.time())
        self.predictions.append(prediction)
        
        # Log detailed info
        logging.info(f"Request processed: latency={latency_ms:.2f}ms, status={status}")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        CPU_USAGE.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        MEMORY_USAGE.set(memory.used)
        
        # GPU metrics
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            GPU_MEMORY_USAGE.set(gpu_memory)
    
    def calculate_throughput(self, window_seconds=60):
        """Calculate requests per second"""
        current_time = time.time()
        recent_requests = [
            t for t in self.request_times 
            if current_time - t <= window_seconds
        ]
        return len(recent_requests) / window_seconds
    
    def update_model_metrics(self, ground_truth_batch=None):
        """Update model-specific metrics"""
        if ground_truth_batch is not None and self.predictions:
            # Calculate accuracy (simplified)
            recent_predictions = self.predictions[-len(ground_truth_batch):]
            if len(recent_predictions) == len(ground_truth_batch):
                accuracy = sum(
                    1 for p, gt in zip(recent_predictions, ground_truth_batch)
                    if abs(p - gt) < 0.1  # Threshold-based accuracy
                ) / len(ground_truth_batch)
                
                MODEL_ACCURACY.set(accuracy)

# Custom metrics collector
class MetricsCollector:
    def __init__(self):
        self.metrics = {}
        self.counters = {}
    
    def increment_counter(self, metric_name: str, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        key = f"{metric_name}:{labels}" if labels else metric_name
        self.counters[key] = self.counters.get(key, 0) + 1
    
    def record_value(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a value for a metric"""
        key = f"{metric_name}:{labels}" if labels else metric_name
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append({
            'value': value,
            'timestamp': time.time()
        })
    
    def get_metric_summary(self, metric_name: str, window_seconds: int = 300):
        """Get summary statistics for a metric"""
        current_time = time.time()
        values = []
        
        for key, metric_data in self.metrics.items():
            if key.startswith(metric_name):
                values.extend([
                    d['value'] for d in metric_data
                    if current_time - d['timestamp'] <= window_seconds
                ])
        
        if not values:
            return None
        
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'p95': sorted(values)[int(0.95 * len(values))] if len(values) > 20 else max(values)
        }

# Health check endpoint
from fastapi import FastAPI, HTTPException

health_app = FastAPI()

@health_app.get("/health")
async def health_check():
    """Basic health check"""
    try:
        # Check model availability
        model_loaded = hasattr(monitor.model, 'forward')
        
        # Check system resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # Check GPU if available
        gpu_available = torch.cuda.is_available()
        gpu_memory_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated() if gpu_available else 0
        
        health_status = {
            "status": "healthy",
            "model_loaded": model_loaded,
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "gpu_available": gpu_available,
            "gpu_memory_free_mb": gpu_memory_free / 1024 / 1024 if gpu_available else 0,
            "timestamp": time.time()
        }
        
        # Check if system is overloaded
        if cpu_percent > 90 or memory.percent > 95:
            health_status["status"] = "degraded"
        
        return health_status
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@health_app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Test model inference
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = monitor.model(test_input)
        
        return {"status": "ready", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Alerting System</Title>
                <Code block language="python">{`import smtplib
import requests
from email.mime.text import MimeText
from dataclasses import dataclass
from typing import List, Callable
import asyncio

@dataclass
class Alert:
    severity: str  # critical, warning, info
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float

class AlertManager:
    def __init__(self):
        self.alert_rules = []
        self.notification_channels = []
        self.active_alerts = {}
        self.alert_cooldown = 300  # 5 minutes
    
    def add_rule(self, metric_name: str, threshold: float, 
                 comparison: str = "greater", severity: str = "warning"):
        """Add alerting rule"""
        rule = {
            'metric_name': metric_name,
            'threshold': threshold,
            'comparison': comparison,
            'severity': severity
        }
        self.alert_rules.append(rule)
    
    def add_notification_channel(self, channel_type: str, config: dict):
        """Add notification channel"""
        if channel_type == "email":
            self.notification_channels.append(EmailNotifier(config))
        elif channel_type == "slack":
            self.notification_channels.append(SlackNotifier(config))
        elif channel_type == "webhook":
            self.notification_channels.append(WebhookNotifier(config))
    
    def evaluate_rules(self, metrics: dict):
        """Evaluate alerting rules against current metrics"""
        current_time = time.time()
        
        for rule in self.alert_rules:
            metric_name = rule['metric_name']
            if metric_name not in metrics:
                continue
            
            current_value = metrics[metric_name]
            threshold = rule['threshold']
            comparison = rule['comparison']
            
            # Evaluate condition
            triggered = False
            if comparison == "greater" and current_value > threshold:
                triggered = True
            elif comparison == "less" and current_value < threshold:
                triggered = True
            elif comparison == "equal" and abs(current_value - threshold) < 0.001:
                triggered = True
            
            alert_key = f"{metric_name}:{comparison}:{threshold}"
            
            if triggered:
                # Check if alert is already active and within cooldown
                if alert_key in self.active_alerts:
                    last_sent = self.active_alerts[alert_key]['last_sent']
                    if current_time - last_sent < self.alert_cooldown:
                        continue
                
                # Create and send alert
                alert = Alert(
                    severity=rule['severity'],
                    title=f"Alert: {metric_name}",
                    description=f"{metric_name} is {current_value} (threshold: {threshold})",
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    timestamp=current_time
                )
                
                self.send_alert(alert)
                self.active_alerts[alert_key] = {
                    'alert': alert,
                    'last_sent': current_time
                }
            else:
                # Clear resolved alerts
                if alert_key in self.active_alerts:
                    self.send_resolution(self.active_alerts[alert_key]['alert'])
                    del self.active_alerts[alert_key]
    
    def send_alert(self, alert: Alert):
        """Send alert to all notification channels"""
        for channel in self.notification_channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                logging.error(f"Failed to send alert via {channel}: {e}")
    
    def send_resolution(self, alert: Alert):
        """Send alert resolution notification"""
        for channel in self.notification_channels:
            try:
                channel.send_resolution(alert)
            except Exception as e:
                logging.error(f"Failed to send resolution via {channel}: {e}")

class EmailNotifier:
    def __init__(self, config: dict):
        self.smtp_server = config['smtp_server']
        self.smtp_port = config['smtp_port']
        self.username = config['username']
        self.password = config['password']
        self.recipients = config['recipients']
    
    def send_alert(self, alert: Alert):
        """Send alert via email"""
        subject = f"[{alert.severity.upper()}] {alert.title}"
        body = f"""
        Alert Details:
        - Metric: {alert.metric_name}
        - Current Value: {alert.current_value}
        - Threshold: {alert.threshold}
        - Severity: {alert.severity}
        - Description: {alert.description}
        - Timestamp: {time.ctime(alert.timestamp)}
        """
        
        msg = MimeText(body)
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = ', '.join(self.recipients)
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)

class SlackNotifier:
    def __init__(self, config: dict):
        self.webhook_url = config['webhook_url']
        self.channel = config.get('channel', '#alerts')
    
    def send_alert(self, alert: Alert):
        """Send alert to Slack"""
        color_map = {"critical": "danger", "warning": "warning", "info": "good"}
        color = color_map.get(alert.severity, "warning")
        
        payload = {
            "channel": self.channel,
            "username": "ML Monitor",
            "icon_emoji": ":warning:",
            "attachments": [{
                "color": color,
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {"title": "Metric", "value": alert.metric_name, "short": True},
                    {"title": "Value", "value": str(alert.current_value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True},
                    {"title": "Severity", "value": alert.severity, "short": True}
                ],
                "ts": int(alert.timestamp)
            }]
        }
        
        response = requests.post(self.webhook_url, json=payload)
        response.raise_for_status()`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Data Drift Detection */}
        <div data-slide className="min-h-[500px]" id="data-drift-detection">
          <Title order={2} mb="xl">Data Drift Detection</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Statistical Drift Detection</Title>
                <Code block language="python">{`import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

class DataDriftDetector:
    def __init__(self, reference_data, confidence_level=0.05):
        self.reference_data = reference_data
        self.confidence_level = confidence_level
        self.reference_stats = self._compute_statistics(reference_data)
        
    def _compute_statistics(self, data):
        """Compute reference statistics"""
        stats_dict = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0)
        }
        return stats_dict
    
    def detect_drift_ks_test(self, new_data):
        """Detect drift using Kolmogorov-Smirnov test"""
        drift_results = {}
        
        for feature_idx in range(self.reference_data.shape[1]):
            reference_feature = self.reference_data[:, feature_idx]
            new_feature = new_data[:, feature_idx]
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(reference_feature, new_feature)
            
            drift_results[f'feature_{feature_idx}'] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'drift_detected': p_value < self.confidence_level
            }
        
        return drift_results
    
    def detect_drift_psi(self, new_data, bins=10):
        """Detect drift using Population Stability Index (PSI)"""
        drift_results = {}
        
        for feature_idx in range(self.reference_data.shape[1]):
            reference_feature = self.reference_data[:, feature_idx]
            new_feature = new_data[:, feature_idx]
            
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference_feature, bins=bins)
            
            # Calculate distributions
            ref_dist, _ = np.histogram(reference_feature, bins=bin_edges)
            new_dist, _ = np.histogram(new_feature, bins=bin_edges)
            
            # Normalize to get proportions
            ref_props = ref_dist / np.sum(ref_dist)
            new_props = new_dist / np.sum(new_dist)
            
            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            new_props = np.where(new_props == 0, 0.0001, new_props)
            
            # Calculate PSI
            psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
            
            # Interpret PSI
            if psi < 0.1:
                stability = "stable"
            elif psi < 0.2:
                stability = "slightly_unstable"
            else:
                stability = "unstable"
            
            drift_results[f'feature_{feature_idx}'] = {
                'psi': psi,
                'stability': stability,
                'drift_detected': psi > 0.2
            }
        
        return drift_results
    
    def detect_drift_wasserstein(self, new_data):
        """Detect drift using Wasserstein distance"""
        drift_results = {}
        
        for feature_idx in range(self.reference_data.shape[1]):
            reference_feature = self.reference_data[:, feature_idx]
            new_feature = new_data[:, feature_idx]
            
            # Calculate Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(reference_feature, new_feature)
            
            # Normalize by feature range for comparison
            feature_range = np.max(reference_feature) - np.min(reference_feature)
            normalized_distance = wasserstein_dist / feature_range if feature_range > 0 else 0
            
            drift_results[f'feature_{feature_idx}'] = {
                'wasserstein_distance': wasserstein_dist,
                'normalized_distance': normalized_distance,
                'drift_detected': normalized_distance > 0.1  # Threshold
            }
        
        return drift_results

# Advanced drift detection with ML models
class MLDriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.classifier = None
        self._train_detector()
    
    def _train_detector(self):
        """Train a classifier to detect drift"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create labels (0 for reference, 1 for new data)
        ref_labels = np.zeros(len(self.reference_data))
        
        # We'll use this to detect drift when new data comes
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        # Initial training with reference data only
        # In practice, you'd retrain when you get new data samples
    
    def detect_drift(self, new_data):
        """Detect drift using trained classifier"""
        # Combine reference and new data
        all_data = np.vstack([self.reference_data, new_data])
        labels = np.hstack([
            np.zeros(len(self.reference_data)), 
            np.ones(len(new_data))
        ])
        
        # Train classifier
        self.classifier.fit(all_data, labels)
        
        # Get prediction probabilities
        probas = self.classifier.predict_proba(all_data)
        
        # Calculate AUC-ROC score
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(labels, probas[:, 1])
        
        # If AUC significantly different from 0.5, there's drift
        drift_detected = auc_score > 0.75 or auc_score < 0.25
        
        return {
            'auc_score': auc_score,
            'drift_detected': drift_detected,
            'feature_importance': self.classifier.feature_importances_.tolist()
        }`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} mb="sm">Real-time Drift Monitoring</Title>
                <Code block language="python">{`import asyncio
from collections import deque
from datetime import datetime, timedelta
import pandas as pd

class RealTimeDriftMonitor:
    def __init__(self, reference_data, window_size=1000, check_interval=3600):
        self.reference_data = reference_data
        self.window_size = window_size
        self.check_interval = check_interval  # seconds
        
        self.drift_detector = DataDriftDetector(reference_data)
        self.data_buffer = deque(maxlen=window_size)
        self.drift_history = []
        
        # Alert thresholds
        self.drift_threshold = 0.05  # p-value threshold
        self.consecutive_alerts = 0
        self.max_consecutive_alerts = 3
        
    def add_data_point(self, data_point):
        """Add new data point to monitoring buffer"""
        self.data_buffer.append({
            'data': data_point,
            'timestamp': datetime.now()
        })
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                if len(self.data_buffer) >= self.window_size:
                    await self.check_for_drift()
                
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Drift monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def check_for_drift(self):
        """Check current window for data drift"""
        if len(self.data_buffer) < 100:  # Need minimum samples
            return
        
        # Extract data from buffer
        current_window = np.array([item['data'] for item in self.data_buffer])
        
        # Detect drift using multiple methods
        ks_results = self.drift_detector.detect_drift_ks_test(current_window)
        psi_results = self.drift_detector.detect_drift_psi(current_window)
        
        # Analyze results
        drift_detected = self.analyze_drift_results(ks_results, psi_results)
        
        # Store results
        drift_result = {
            'timestamp': datetime.now(),
            'drift_detected': drift_detected,
            'ks_results': ks_results,
            'psi_results': psi_results,
            'window_size': len(current_window)
        }
        
        self.drift_history.append(drift_result)
        
        # Handle alerts
        if drift_detected:
            self.consecutive_alerts += 1
            if self.consecutive_alerts >= self.max_consecutive_alerts:
                await self.trigger_drift_alert(drift_result)
        else:
            self.consecutive_alerts = 0
    
    def analyze_drift_results(self, ks_results, psi_results):
        """Analyze drift detection results"""
        # Count features with detected drift
        ks_drift_count = sum(1 for result in ks_results.values() if result['drift_detected'])
        psi_drift_count = sum(1 for result in psi_results.values() if result['drift_detected'])
        
        total_features = len(ks_results)
        
        # Consider drift detected if >30% of features show drift in either test
        ks_drift_ratio = ks_drift_count / total_features
        psi_drift_ratio = psi_drift_count / total_features
        
        return ks_drift_ratio > 0.3 or psi_drift_ratio > 0.3
    
    async def trigger_drift_alert(self, drift_result):
        """Trigger drift alert"""
        alert = Alert(
            severity="warning",
            title="Data Drift Detected",
            description=f"Significant data drift detected in {drift_result['window_size']} recent samples",
            metric_name="data_drift",
            current_value=1.0,
            threshold=0.3,
            timestamp=time.time()
        )
        
        # Reset consecutive alerts counter
        self.consecutive_alerts = 0
        
        # Log drift detection
        logging.warning(f"Data drift detected: {drift_result}")
        
        # You would send this to your alerting system
        # alert_manager.send_alert(alert)
    
    def get_drift_summary(self, hours=24):
        """Get drift summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_history = [
            h for h in self.drift_history 
            if h['timestamp'] > cutoff_time
        ]
        
        if not recent_history:
            return {"status": "no_data", "period_hours": hours}
        
        drift_count = sum(1 for h in recent_history if h['drift_detected'])
        total_checks = len(recent_history)
        
        return {
            "status": "drift_detected" if drift_count > 0 else "stable",
            "period_hours": hours,
            "total_checks": total_checks,
            "drift_detected_count": drift_count,
            "drift_rate": drift_count / total_checks if total_checks > 0 else 0,
            "last_check": recent_history[-1]['timestamp'].isoformat()
        }

# Model performance monitoring
class ModelPerformanceMonitor:
    def __init__(self, model, validation_data):
        self.model = model
        self.validation_data = validation_data
        self.baseline_metrics = self._compute_baseline_metrics()
        self.performance_history = deque(maxlen=1000)
    
    def _compute_baseline_metrics(self):
        """Compute baseline performance metrics"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in self.validation_data:
                inputs, labels = batch
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def evaluate_current_performance(self):
        """Evaluate current model performance"""
        current_metrics = self._compute_baseline_metrics()
        
        # Calculate degradation
        degradation = {}
        for metric, baseline_value in self.baseline_metrics.items():
            current_value = current_metrics[metric]
            degradation[metric] = (baseline_value - current_value) / baseline_value
        
        # Store in history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': current_metrics,
            'degradation': degradation
        })
        
        return current_metrics, degradation
    
    def check_performance_degradation(self, threshold=0.1):
        """Check if model performance has degraded significantly"""
        if not self.performance_history:
            return False
        
        recent_performance = self.performance_history[-1]
        
        # Check if any metric degraded more than threshold
        for metric, degradation in recent_performance['degradation'].items():
            if degradation > threshold:
                logging.warning(f"Performance degradation detected in {metric}: {degradation:.3f}")
                return True
        
        return False`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: Logging and Observability */}
        <div data-slide className="min-h-[500px]" id="logging-observability">
          <Title order={2} mb="xl">Logging and Observability</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">Structured Logging</Title>
                <Code block language="python">{`import logging
import json
from datetime import datetime
from typing import Dict, Any
import traceback

class StructuredLogger:
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create structured formatter
        formatter = self.StructuredFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logs
        file_handler = logging.FileHandler('model_service.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add extra fields if present
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': str(record.exc_info[1]),
                    'traceback': traceback.format_exception(*record.exc_info)
                }
            
            return json.dumps(log_entry)
    
    def log_prediction(self, request_id: str, model_version: str, 
                      input_shape: tuple, prediction: Any, 
                      latency_ms: float, confidence: float = None):
        """Log model prediction with context"""
        extra_fields = {
            'event_type': 'prediction',
            'request_id': request_id,
            'model_version': model_version,
            'input_shape': input_shape,
            'output_type': type(prediction).__name__,
            'latency_ms': latency_ms,
            'confidence': confidence
        }
        
        self.logger.info(
            f"Prediction completed for request {request_id}",
            extra={'extra_fields': extra_fields}
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        extra_fields = {
            'event_type': 'error',
            'error_type': type(error).__name__,
            'context': context or {}
        }
        
        self.logger.error(
            f"Error occurred: {str(error)}",
            extra={'extra_fields': extra_fields},
            exc_info=True
        )
    
    def log_model_update(self, old_version: str, new_version: str, 
                        update_type: str, metrics: Dict[str, float] = None):
        """Log model updates"""
        extra_fields = {
            'event_type': 'model_update',
            'old_version': old_version,
            'new_version': new_version,
            'update_type': update_type,
            'metrics': metrics or {}
        }
        
        self.logger.info(
            f"Model updated from {old_version} to {new_version}",
            extra={'extra_fields': extra_fields}
        )

# Request tracing
class RequestTracer:
    def __init__(self):
        self.active_requests = {}
    
    def start_request(self, request_id: str, endpoint: str, user_id: str = None):
        """Start tracing a request"""
        self.active_requests[request_id] = {
            'request_id': request_id,
            'endpoint': endpoint,
            'user_id': user_id,
            'start_time': time.time(),
            'events': []
        }
    
    def add_event(self, request_id: str, event_name: str, data: Dict[str, Any] = None):
        """Add event to request trace"""
        if request_id in self.active_requests:
            event = {
                'event_name': event_name,
                'timestamp': time.time(),
                'data': data or {}
            }
            self.active_requests[request_id]['events'].append(event)
    
    def finish_request(self, request_id: str, status: str = "success"):
        """Finish request tracing"""
        if request_id in self.active_requests:
            request_trace = self.active_requests[request_id]
            request_trace['status'] = status
            request_trace['duration'] = time.time() - request_trace['start_time']
            
            # Log complete trace
            logger.info(
                f"Request completed: {request_id}",
                extra={'extra_fields': request_trace}
            )
            
            # Clean up
            del self.active_requests[request_id]
            
            return request_trace

# Distributed tracing with OpenTelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing():
    """Setup distributed tracing"""
    # Configure tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=14268,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return tracer

def trace_prediction(tracer, model_function):
    """Decorator to trace model predictions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span("model_prediction") as span:
                # Add attributes
                span.set_attribute("model.name", "default_model")
                span.set_attribute("input.shape", str(args[0].shape) if args else "unknown")
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("prediction.status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("prediction.status", "error")
                    span.set_attribute("error.message", str(e))
                    raise
        return wrapper
    return decorator`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} mb="sm">Dashboard and Visualization</Title>
                <Code block language="python">{`import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, dcc, html, callback, Output, Input
import pandas as pd
from datetime import datetime, timedelta

class MonitoringDashboard:
    def __init__(self, metrics_collector, drift_monitor):
        self.metrics_collector = metrics_collector
        self.drift_monitor = drift_monitor
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("ML Model Monitoring Dashboard"),
            
            # Key metrics cards
            html.Div([
                html.Div([
                    html.H3("Requests/sec"),
                    html.H2(id="requests-per-sec", children="0")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Avg Latency"),
                    html.H2(id="avg-latency", children="0ms")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Model Accuracy"),
                    html.H2(id="model-accuracy", children="0%")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Data Drift Status"),
                    html.H2(id="drift-status", children="Stable")
                ], className="metric-card")
            ], className="metrics-row"),
            
            # Time series charts
            html.Div([
                dcc.Graph(id="latency-chart"),
                dcc.Graph(id="throughput-chart")
            ], className="charts-row"),
            
            html.Div([
                dcc.Graph(id="error-rate-chart"),
                dcc.Graph(id="drift-score-chart")
            ], className="charts-row"),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=10*1000,  # Update every 10 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @callback(
            [Output('requests-per-sec', 'children'),
             Output('avg-latency', 'children'),
             Output('model-accuracy', 'children'),
             Output('drift-status', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            # Get current metrics
            throughput = self.metrics_collector.calculate_throughput()
            latency_summary = self.metrics_collector.get_metric_summary('request_latency')
            accuracy_summary = self.metrics_collector.get_metric_summary('model_accuracy')
            drift_summary = self.drift_monitor.get_drift_summary(hours=1)
            
            # Format values
            rps = f"{throughput:.1f}"
            avg_latency = f"{latency_summary['mean']:.1f}ms" if latency_summary else "N/A"
            accuracy = f"{accuracy_summary['mean']*100:.1f}%" if accuracy_summary else "N/A"
            drift_status = drift_summary.get('status', 'Unknown').title()
            
            return rps, avg_latency, accuracy, drift_status
        
        @callback(
            Output('latency-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_latency_chart(n):
            # Get latency data for the last hour
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            # This would typically come from your metrics database
            latency_data = self.get_time_series_data('request_latency', start_time, end_time)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=latency_data['timestamp'],
                y=latency_data['value'],
                mode='lines',
                name='Latency (ms)',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title='Request Latency Over Time',
                xaxis_title='Time',
                yaxis_title='Latency (ms)',
                height=300
            )
            
            return fig
        
        @callback(
            Output('drift-score-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_drift_chart(n):
            # Get drift scores
            drift_history = self.drift_monitor.drift_history[-100:]  # Last 100 checks
            
            if not drift_history:
                return go.Figure()
            
            timestamps = [h['timestamp'] for h in drift_history]
            
            # Extract average PSI scores across features
            psi_scores = []
            for h in drift_history:
                psi_values = [r['psi'] for r in h['psi_results'].values()]
                psi_scores.append(np.mean(psi_values))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=psi_scores,
                mode='lines+markers',
                name='PSI Score',
                line=dict(color='red')
            ))
            
            # Add threshold line
            fig.add_hline(y=0.2, line_dash="dash", line_color="orange", 
                         annotation_text="Warning Threshold")
            
            fig.update_layout(
                title='Data Drift Score Over Time',
                xaxis_title='Time',
                yaxis_title='PSI Score',
                height=300
            )
            
            return fig
    
    def get_time_series_data(self, metric_name, start_time, end_time):
        """Get time series data for a metric"""
        # This would typically query your metrics database
        # For demo, return sample data
        timestamps = pd.date_range(start_time, end_time, freq='1min')
        values = np.random.normal(100, 20, len(timestamps))  # Sample latency data
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
    
    def run(self, host='0.0.0.0', port=8050):
        """Run the dashboard"""
        self.app.run_server(host=host, port=port, debug=False)

# Log analysis and aggregation
class LogAnalyzer:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
    
    def analyze_error_patterns(self, time_window_hours=24):
        """Analyze error patterns in logs"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        errors = []
        with open(self.log_file_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    
                    if (log_entry.get('level') == 'ERROR' and 
                        datetime.fromisoformat(log_entry['timestamp']) > cutoff_time):
                        errors.append(log_entry)
                except json.JSONDecodeError:
                    continue
        
        # Analyze error patterns
        error_types = {}
        for error in errors:
            error_type = error.get('extra_fields', {}).get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(errors),
            'error_types': error_types,
            'error_rate_per_hour': len(errors) / time_window_hours
        }
    
    def get_performance_metrics(self, time_window_hours=1):
        """Extract performance metrics from logs"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        latencies = []
        request_count = 0
        
        with open(self.log_file_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    
                    if (log_entry.get('extra_fields', {}).get('event_type') == 'prediction' and
                        datetime.fromisoformat(log_entry['timestamp']) > cutoff_time):
                        
                        latency = log_entry['extra_fields'].get('latency_ms')
                        if latency:
                            latencies.append(latency)
                            request_count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if not latencies:
            return None
        
        return {
            'request_count': request_count,
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_per_hour': request_count / time_window_hours
        }`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 5: Automated Model Updates */}
        <div data-slide className="min-h-[500px]" id="automated-updates">
          <Title order={2} mb="xl">Automated Model Updates</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-indigo-50 mb-4">
                <Title order={4} mb="sm">Automated Retraining Pipeline</Title>
                <Code block language="python">{`import asyncio
import torch
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os
import tempfile
import shutil

class AutomatedRetrainingPipeline:
    def __init__(self, model_trainer, data_collector, model_registry, 
                 validation_data, retraining_config):
        self.model_trainer = model_trainer
        self.data_collector = data_collector
        self.model_registry = model_registry
        self.validation_data = validation_data
        self.config = retraining_config
        
        # Retraining triggers
        self.performance_threshold = retraining_config.get('performance_threshold', 0.95)
        self.drift_threshold = retraining_config.get('drift_threshold', 0.2)
        self.data_freshness_days = retraining_config.get('data_freshness_days', 30)
        self.min_new_samples = retraining_config.get('min_new_samples', 1000)
        
        # Schedule
        self.check_interval_hours = retraining_config.get('check_interval_hours', 24)
        
        # Current model info
        self.current_model_version = None
        self.current_model_performance = None
    
    async def monitoring_loop(self):
        """Main monitoring and retraining loop"""
        while True:
            try:
                should_retrain, reason = await self.should_trigger_retraining()
                
                if should_retrain:
                    logger.info(f"Triggering retraining: {reason}")
                    await self.execute_retraining_pipeline(reason)
                else:
                    logger.info("Retraining not needed at this time")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in retraining loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def should_trigger_retraining(self) -> tuple[bool, str]:
        """Determine if retraining should be triggered"""
        reasons = []
        
        # Check model performance degradation
        current_performance = await self.evaluate_current_model()
        if (self.current_model_performance and 
            current_performance['accuracy'] < self.current_model_performance['accuracy'] * self.performance_threshold):
            reasons.append(f"Performance degradation: {current_performance['accuracy']:.3f} < {self.current_model_performance['accuracy'] * self.performance_threshold:.3f}")
        
        # Check data drift
        drift_detected = await self.check_data_drift()
        if drift_detected:
            reasons.append("Significant data drift detected")
        
        # Check data freshness
        last_training_date = await self.get_last_training_date()
        if last_training_date and (datetime.now() - last_training_date).days > self.data_freshness_days:
            reasons.append(f"Data freshness: {(datetime.now() - last_training_date).days} days since last training")
        
        # Check new data availability
        new_sample_count = await self.count_new_samples()
        if new_sample_count >= self.min_new_samples:
            reasons.append(f"New data available: {new_sample_count} samples")
        
        return len(reasons) > 0, "; ".join(reasons)
    
    async def execute_retraining_pipeline(self, trigger_reason: str):
        """Execute the complete retraining pipeline"""
        pipeline_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting retraining pipeline {pipeline_id}")
            
            # Step 1: Collect and prepare data
            training_data = await self.prepare_training_data()
            logger.info(f"Prepared training data: {len(training_data)} samples")
            
            # Step 2: Train new model
            new_model, training_metrics = await self.train_new_model(training_data)
            logger.info(f"Training completed with metrics: {training_metrics}")
            
            # Step 3: Validate new model
            validation_results = await self.validate_new_model(new_model)
            logger.info(f"Validation results: {validation_results}")
            
            # Step 4: Compare with current model
            comparison_results = await self.compare_models(new_model, validation_results)
            
            # Step 5: Deploy if better
            if comparison_results['deploy_new_model']:
                await self.deploy_new_model(new_model, pipeline_id, {
                    'trigger_reason': trigger_reason,
                    'training_metrics': training_metrics,
                    'validation_results': validation_results,
                    'comparison_results': comparison_results
                })
                logger.info(f"Successfully deployed new model from pipeline {pipeline_id}")
            else:
                logger.info(f"New model not deployed: {comparison_results['reason']}")
            
        except Exception as e:
            logger.error(f"Retraining pipeline {pipeline_id} failed: {e}")
            await self.handle_pipeline_failure(pipeline_id, e)
    
    async def prepare_training_data(self):
        """Collect and prepare training data"""
        # Get recent data
        recent_data = await self.data_collector.get_recent_data(
            days=self.data_freshness_days
        )
        
        # Get historical data for baseline
        historical_data = await self.data_collector.get_historical_data(
            start_date=datetime.now() - timedelta(days=365),
            limit=50000  # Limit to manage memory
        )
        
        # Combine and balance data
        combined_data = self.balance_dataset(recent_data, historical_data)
        
        # Apply data quality checks
        cleaned_data = await self.data_quality_checks(combined_data)
        
        return cleaned_data
    
    async def train_new_model(self, training_data):
        """Train a new model with the prepared data"""
        # Create temporary directory for training
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save training data
            train_data_path = os.path.join(temp_dir, 'train_data.pt')
            torch.save(training_data, train_data_path)
            
            # Configure training
            training_config = {
                'epochs': self.config.get('training_epochs', 10),
                'batch_size': self.config.get('batch_size', 32),
                'learning_rate': self.config.get('learning_rate', 0.001),
                'data_path': train_data_path,
                'output_dir': temp_dir
            }
            
            # Train model
            model, metrics = await self.model_trainer.train_async(training_config)
            
            return model, metrics
    
    async def validate_new_model(self, model):
        """Validate the new model"""
        model.eval()
        
        validation_metrics = {}
        
        # Basic accuracy metrics
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.validation_data:
                inputs, labels = batch
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        validation_metrics['accuracy'] = correct / total
        
        # Additional metrics
        validation_metrics['latency'] = await self.benchmark_model_latency(model)
        validation_metrics['memory_usage'] = self.calculate_model_memory_usage(model)
        
        return validation_metrics
    
    async def compare_models(self, new_model, new_metrics):
        """Compare new model with current model"""
        if not self.current_model_performance:
            return {'deploy_new_model': True, 'reason': 'No current model for comparison'}
        
        current_accuracy = self.current_model_performance['accuracy']
        new_accuracy = new_metrics['accuracy']
        
        # Define minimum improvement threshold
        min_improvement = self.config.get('min_improvement_threshold', 0.01)
        
        if new_accuracy > current_accuracy + min_improvement:
            return {
                'deploy_new_model': True,
                'reason': f'Accuracy improved: {new_accuracy:.3f} > {current_accuracy:.3f}',
                'improvement': new_accuracy - current_accuracy
            }
        elif new_accuracy < current_accuracy - min_improvement:
            return {
                'deploy_new_model': False,
                'reason': f'Accuracy degraded: {new_accuracy:.3f} < {current_accuracy:.3f}',
                'degradation': current_accuracy - new_accuracy
            }
        else:
            # Check other factors like latency, memory usage
            current_latency = self.current_model_performance.get('latency', float('inf'))
            new_latency = new_metrics.get('latency', float('inf'))
            
            if new_latency < current_latency * 0.9:  # 10% improvement in latency
                return {
                    'deploy_new_model': True,
                    'reason': f'Latency improved: {new_latency:.3f}ms < {current_latency:.3f}ms'
                }
            
            return {
                'deploy_new_model': False,
                'reason': 'No significant improvement detected'
            }
    
    async def deploy_new_model(self, model, pipeline_id, metadata):
        """Deploy the new model to production"""
        # Create new model version
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model to registry
        model_path = await self.model_registry.save_model(model, version, metadata)
        
        # Update model server (this would typically involve a rolling update)
        await self.update_model_server(model_path, version)
        
        # Update current model tracking
        self.current_model_version = version
        self.current_model_performance = metadata['validation_results']
        
        logger.info(f"Model deployment completed: {version}")
    
    async def update_model_server(self, model_path, version):
        """Update the model server with new model"""
        # This would typically involve:
        # 1. Uploading model to model server
        # 2. Triggering rolling update in Kubernetes
        # 3. Validating the deployment
        
        # For example, using kubectl
        update_command = f"""
        kubectl set image deployment/ml-model-server \\
            ml-model=your-registry/ml-model:{version} \\
            --namespace=production
        """
        
        # Wait for rollout to complete
        rollout_status_command = """
        kubectl rollout status deployment/ml-model-server --namespace=production
        """
        
        # In practice, you'd execute these commands or use the Kubernetes Python client
        logger.info(f"Triggered model server update to {version}")

# Configuration management
class RetrainingConfig:
    def __init__(self):
        self.config = {
            'performance_threshold': float(os.environ.get('PERFORMANCE_THRESHOLD', '0.95')),
            'drift_threshold': float(os.environ.get('DRIFT_THRESHOLD', '0.2')),
            'data_freshness_days': int(os.environ.get('DATA_FRESHNESS_DAYS', '30')),
            'min_new_samples': int(os.environ.get('MIN_NEW_SAMPLES', '1000')),
            'check_interval_hours': int(os.environ.get('CHECK_INTERVAL_HOURS', '24')),
            'training_epochs': int(os.environ.get('TRAINING_EPOCHS', '10')),
            'batch_size': int(os.environ.get('BATCH_SIZE', '32')),
            'learning_rate': float(os.environ.get('LEARNING_RATE', '0.001')),
            'min_improvement_threshold': float(os.environ.get('MIN_IMPROVEMENT_THRESHOLD', '0.01'))
        }
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration dynamically"""
        self.config.update(updates)
        logger.info(f"Configuration updated: {updates}")`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 6: Best Practices Summary */}
        <div data-slide className="min-h-[500px]" id="best-practices">
          <Title order={2} mb="xl">Monitoring Best Practices</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Key Principles</Title>
                <List spacing="sm">
                  <List.Item><strong>Observability by Design:</strong> Build monitoring into your ML systems from the start</List.Item>
                  <List.Item><strong>Multi-layer Monitoring:</strong> Monitor infrastructure, application, and ML-specific metrics</List.Item>
                  <List.Item><strong>Proactive Alerting:</strong> Set up alerts for leading indicators, not just failures</List.Item>
                  <List.Item><strong>Automated Response:</strong> Implement automated remediation for common issues</List.Item>
                  <List.Item><strong>Continuous Validation:</strong> Continuously validate model assumptions and performance</List.Item>
                  <List.Item><strong>Data Quality Focus:</strong> Prioritize data quality monitoring over model metrics</List.Item>
                </List>
                
                <Paper className="p-3 bg-white mt-4">
                  <Title order={5} className="mb-2">Monitoring Checklist</Title>
                  <List size="sm">
                    <List.Item>✅ Request latency and throughput</List.Item>
                    <List.Item>✅ Model accuracy and confidence</List.Item>
                    <List.Item>✅ Data drift detection</List.Item>
                    <List.Item>✅ Resource utilization</List.Item>
                    <List.Item>✅ Error rates and types</List.Item>
                    <List.Item>✅ Feature distribution changes</List.Item>
                    <List.Item>✅ Model version tracking</List.Item>
                    <List.Item>✅ Business metrics correlation</List.Item>
                  </List>
                </Paper>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Common Pitfalls to Avoid</Title>
                <List spacing="sm">
                  <List.Item><strong>Alert Fatigue:</strong> Too many false positive alerts reduce responsiveness</List.Item>
                  <List.Item><strong>Vanity Metrics:</strong> Focusing on metrics that don't drive business value</List.Item>
                  <List.Item><strong>Delayed Detection:</strong> Not monitoring leading indicators of problems</List.Item>
                  <List.Item><strong>Manual Processes:</strong> Relying on manual intervention for routine issues</List.Item>
                  <List.Item><strong>Siloed Monitoring:</strong> Not connecting ML metrics to business outcomes</List.Item>
                  <List.Item><strong>Inadequate Baselines:</strong> Not establishing proper baseline metrics</List.Item>
                </List>
                
                <Paper className="p-3 bg-white mt-4">
                  <Title order={5} className="mb-2">Maintenance Schedule</Title>
                  <List size="sm">
                    <List.Item><strong>Daily:</strong> Check system health and recent alerts</List.Item>
                    <List.Item><strong>Weekly:</strong> Review performance trends and drift metrics</List.Item>
                    <List.Item><strong>Monthly:</strong> Analyze model degradation and retrain if needed</List.Item>
                    <List.Item><strong>Quarterly:</strong> Review and update monitoring thresholds</List.Item>
                    <List.Item><strong>Yearly:</strong> Audit entire monitoring strategy</List.Item>
                  </List>
                </Paper>
                
                <Paper className="p-3 bg-gray-100 mt-4">
                  <Title order={5} className="mb-2">Tools and Technologies</Title>
                  <List size="sm">
                    <List.Item>📊 <strong>Metrics:</strong> Prometheus, Grafana, DataDog</List.Item>
                    <List.Item>📝 <strong>Logging:</strong> ELK Stack, Fluentd, Loki</List.Item>
                    <List.Item>🔍 <strong>Tracing:</strong> Jaeger, Zipkin, OpenTelemetry</List.Item>
                    <List.Item>🚨 <strong>Alerting:</strong> PagerDuty, Slack, Email</List.Item>
                    <List.Item>📈 <strong>ML Monitoring:</strong> MLflow, Weights & Biases, Evidently AI</List.Item>
                  </List>
                </Paper>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default MonitoringMaintenance;