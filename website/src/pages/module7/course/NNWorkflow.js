import React from 'react';
import { Stack, Title, Text, List, Box, Grid, Alert, Table } from '@mantine/core';
import { InlineMath } from 'react-katex';
import { Database, Cpu, LineChart, Save, CheckCircle2 } from 'lucide-react';
import CodeBlock from 'components/CodeBlock';

const NNWorkflow = () => {
  return (
    <Stack spacing="xl">
      <Title order={1} id="nn-workflow">Neural Network Training Workflow</Title>
      
      <Text size="lg">
      Just like in classical machine learning, properly prepare the data, train, and eval the neural networks with PyTorch.
      </Text>

      {/* Example Data Generation */}
      <Title order={2} id="example-data">Example Regression Problem</Title>
      
      <Text>
        We'll use a simple nonlinear regression problem : predicting y = 0.2x² + 0.5x + 2 + noise
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch
import numpy as np

# Generate synthetic regression data
X = np.linspace(-5, 5, 1000).reshape(-1, 1)
y = 0.2 * X**2 + 0.5 * X + 2 + np.random.normal(0, 0.2, X.shape)

# Split data
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]`}
      />

      {/* 1. DATA PREPARATION */}
      <Title order={2} id="data-preparation" className="flex items-center gap-2">
        <Database size={24} />
        Data Preparation
      </Title>

      <Grid gutter="lg">
        <Grid.Col span={12}>
          <Box p="md" className="border rounded">
            <Title order={4}>Feature Scaling</Title>
            <Text>
              Scale input features to have zero mean and unit variance using StandardScaler.
            </Text>
            <InlineMath>{`x_{scaled} = \\frac{x - \\mu}{\\sigma}`}</InlineMath>
          </Box>
        </Grid.Col>
      </Grid>

      <CodeBlock
        language="python"
        code={`
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create datasets and dataloaders
train_dataset = RegressionDataset(X_train_scaled, y_train)
val_dataset = RegressionDataset(X_val_scaled, y_val)
test_dataset = RegressionDataset(X_test_scaled, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)`}
      />

      {/* 2. MODEL AND DEVICE */}
      <Title order={2} id="model-device" className="flex items-center gap-2">
        <Cpu size={24} />
        Model and Device Setup
      </Title>

      <CodeBlock
        language="python"
        code={`
import torch.nn as nn

# Simple regression model
class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RegressionNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)`}
      />

      {/* 3. TRAINING */}
      <Title order={2} id="training" className="flex items-center gap-2">
        <LineChart size={24} />
        Training
      </Title>

      <CodeBlock
        language="python"
        code={`
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

# Training loop
epochs = 100
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')`}
      />

      {/* 4. SAVE/LOAD */}
      <Title order={2} id="save-load" className="flex items-center gap-2">
        <Save size={24} />
        Save and Load Model
      </Title>

      <CodeBlock
        language="python"
        code={`
# Save model
torch.save(model.state_dict(), 'regression_model.pth')

# Load model
loaded_model = RegressionNet()
loaded_model.load_state_dict(torch.load('regression_model.pth'))
loaded_model.to(device)`}
      />

      {/* 5. PREDICT */}
      <Title order={2} id="predict" className="flex items-center gap-2">
        <CheckCircle2 size={24} />
        Make Predictions
      </Title>

      <CodeBlock
        language="python"
        code={`
def predict(model, X_new, scaler, device):
    model.eval()
    # Scale new data
    X_scaled = scaler.transform(X_new)
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    with torch.no_grad():
        predictions = model(X_tensor)
    
    return predictions.cpu().numpy()

# Example prediction
X_new = np.array([[2.0], [3.0], [4.0]])
predictions = predict(model, X_new, scaler, device)
print('Predictions:', predictions)`}
      />

      <Box p="lg" className="bg-blue-50 rounded">
        <Title order={4}>Essential Steps Summary</Title>
        <List>
          <List.Item>✓ Prepare data: scale features, create dataloaders</List.Item>
          <List.Item>✓ Define model and move to appropriate device</List.Item>
          <List.Item>✓ Train model with validation</List.Item>
          <List.Item>✓ Save trained model</List.Item>
          <List.Item>✓ Make predictions on new data</List.Item>
        </List>
      </Box>
    </Stack>
  );
};

export default NNWorkflow;