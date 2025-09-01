import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';

const DeploymentStrategies = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Deployment Strategies
          </Title>
          <Text size="xl" className="text-center mb-6">
            From Local Models to Production Systems
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                Learn how to deploy machine learning models in production environments,
                from simple API servers to scalable cloud platforms.
              </Text>
              <List>
                <List.Item>Model serving architectures</List.Item>
                <List.Item>Containerization with Docker</List.Item>
                <List.Item>Cloud deployment strategies</List.Item>
                <List.Item>Monitoring and scaling</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        <div data-slide className="min-h-[500px]" id="serving-models">
          <Title order={2} className="mb-6">Model Serving</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">REST API with Flask</Title>
                <Code block language="python">{`from flask import Flask, request, jsonify
import torch
import pickle

app = Flask(__name__)

# Load model at startup
model = torch.load('model.pt', map_location='cpu')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        inputs = torch.tensor(data['inputs'])
        
        # Make prediction
        with torch.no_grad():
            prediction = model(inputs)
        
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">FastAPI Alternative</Title>
                <Code block language="python">{`from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn

app = FastAPI(title="ML Model API")

class PredictionRequest(BaseModel):
    inputs: list

class PredictionResponse(BaseModel):
    prediction: list
    status: str

# Load model
model = torch.load('model.pt', map_location='cpu')
model.eval()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        inputs = torch.tensor(request.inputs)
        
        with torch.no_grad():
            prediction = model(inputs)
            
        return PredictionResponse(
            prediction=prediction.tolist(),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        <div data-slide className="min-h-[500px]" id="containerization">
          <Title order={2} className="mb-6">Containerization with Docker</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Dockerfile</Title>
                <Code block language="dockerfile">{`FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "app.py"]`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} mb="sm">Docker Commands</Title>
                <Code block language="bash">{`# Build the image
docker build -t ml-model:latest .

# Run locally
docker run -p 8000:8000 ml-model:latest

# Run with environment variables
docker run -p 8000:8000 \\
  -e MODEL_PATH=/app/models/model.pt \\
  -e LOG_LEVEL=INFO \\
  ml-model:latest

# Tag for registry
docker tag ml-model:latest myregistry/ml-model:v1.0

# Push to registry  
docker push myregistry/ml-model:v1.0

# Multi-stage build for smaller images
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        <div data-slide className="min-h-[500px]" id="cloud-deployment">
          <Title order={2} className="mb-6">Cloud Deployment</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={4}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">AWS Deployment</Title>
                <List size="sm" spacing="sm">
                  <List.Item><strong>Amazon ECS:</strong> Container orchestration</List.Item>
                  <List.Item><strong>AWS Lambda:</strong> Serverless functions</List.Item>
                  <List.Item><strong>SageMaker:</strong> ML-specific hosting</List.Item>
                  <List.Item><strong>Elastic Beanstalk:</strong> Platform-as-a-service</List.Item>
                  <List.Item><strong>EKS:</strong> Managed Kubernetes</List.Item>
                </List>
                <Code block language="bash" className="mt-3">{`# Deploy to ECS
aws ecs create-service \\
  --cluster ml-cluster \\
  --service-name ml-model \\
  --task-definition ml-model:1 \\
  --desired-count 2

# Deploy to Lambda (with container)
aws lambda create-function \\
  --function-name ml-model \\
  --package-type Image \\
  --code ImageUri=account.dkr.ecr.region.amazonaws.com/ml-model:latest`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={4}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Google Cloud Platform</Title>
                <List size="sm" spacing="sm">
                  <List.Item><strong>Cloud Run:</strong> Serverless containers</List.Item>
                  <List.Item><strong>GKE:</strong> Managed Kubernetes</List.Item>
                  <List.Item><strong>AI Platform:</strong> ML model serving</List.Item>
                  <List.Item><strong>App Engine:</strong> Platform-as-a-service</List.Item>
                  <List.Item><strong>Cloud Functions:</strong> Serverless</List.Item>
                </List>
                <Code block language="bash" className="mt-3">{`# Deploy to Cloud Run
gcloud run deploy ml-model \\
  --image gcr.io/project/ml-model \\
  --platform managed \\
  --region us-central1 \\
  --allow-unauthenticated

# Deploy to AI Platform
gcloud ai-platform models create ml_model
gcloud ai-platform versions create v1 \\
  --model ml_model \\
  --origin gs://bucket/model/`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={4}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Microsoft Azure</Title>
                <List size="sm" spacing="sm">
                  <List.Item><strong>Container Instances:</strong> Simple containers</List.Item>
                  <List.Item><strong>AKS:</strong> Managed Kubernetes</List.Item>
                  <List.Item><strong>ML Studio:</strong> Model deployment</List.Item>
                  <List.Item><strong>App Service:</strong> Web apps</List.Item>
                  <List.Item><strong>Functions:</strong> Serverless</List.Item>
                </List>
                <Code block language="bash" className="mt-3">{`# Deploy to Container Instances
az container create \\
  --resource-group myResourceGroup \\
  --name ml-model \\
  --image myregistry/ml-model:latest \\
  --ports 80

# Deploy to AKS
kubectl apply -f k8s-deployment.yaml
kubectl expose deployment ml-model \\
  --type=LoadBalancer --port=80`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default DeploymentStrategies;