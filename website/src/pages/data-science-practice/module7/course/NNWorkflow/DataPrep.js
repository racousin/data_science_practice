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
        First, we split our data into training, validation, and test sets.
      </Text>

      <CodeBlock
        language="python"
        code={`
# Split data
from sklearn.model_selection import train_test_split

# Split data into train and remaining (val + test) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the remaining data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
`}
      />

<Grid gutter="lg">
  <Grid.Col span={12}>
    <Box p="md" className="border rounded">
      <Title order={4}>2. Feature Scaling</Title>
      <Text>
        Neural networks are particularly sensitive to unscaled data due to how they learn weights through gradient descent. If input features have vastly different scales, the model's convergence can be slow or unstable. This is because the gradients of large-scale features may dominate, causing the model to make uneven updates, leading to slower learning and potentially suboptimal results.
      </Text>
      <Text>
        To mitigate this issue, scale input features for example with <strong>StandardScaler</strong>.
      </Text>
      <InlineMath>{`x_{scaled} = \\frac{x - \\mu}{\\sigma}`}</InlineMath>
    </Box>
  </Grid.Col>
</Grid>

<CodeBlock
  language="python"
  code={`
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
  `}
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