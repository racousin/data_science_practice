import React from 'react';
import { Stack, Title, Text, Box, Grid } from '@mantine/core';
import { InlineMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import { Link } from 'react-router-dom';

const DataPrep = () => {
  return (
    <Stack spacing="md">
            <Title order={4}>0. Preprocessing Steps</Title>
      <Text>
        All the preprocessing steps we covered in the Machine Learning course remain essential 
        for neural networks:
        <ul>
          <li>Handling Duplicate Data</li>
          <li>Managing Missing Values</li>
          <li>Processing Categorical Variables</li>
          <li>Detecting and Handling Outliers</li>
          <li>Feature Engineering</li>
        </ul>
        Please refer to our <Link to="/module5/course" className="text-blue-500 hover:underline">Data Preprocessing</Link>.
      </Text>
      <Title order={4}>1. Split the Data</Title>
      <Text>
        First, we split our data into training, validation, and test sets using a 70-15-15 split ratio.
      </Text>

      <CodeBlock
        language="python"
        code={`
# Split data
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]`}
      />

      <Grid gutter="lg">
        <Grid.Col span={12}>
          <Box p="md" className="border rounded">
            <Title order={4}>2. Feature Scaling</Title>
            <Text>
              Scale input features to have zero mean and unit variance using StandardScaler:
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
X_test_scaled = scaler.transform(X_test)`}
      />

      <Title order={4}>3. Create DataLoaders</Title>
      <Text>
        PyTorch DataLoaders handle batching, shuffling, and loading the data efficiently during training.
      </Text>

      <CodeBlock
        language="python"
        code={`
# Create datasets
train_dataset = RegressionDataset(X_train_scaled, y_train)
val_dataset = RegressionDataset(X_val_scaled, y_val)
test_dataset = RegressionDataset(X_test_scaled, y_test)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)`}
      />
    </Stack>
  );
};

export default DataPrep;