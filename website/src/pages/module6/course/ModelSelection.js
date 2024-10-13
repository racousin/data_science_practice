import React from 'react';
import { Container, Title, Text, Stack, List, Group, Image } from '@mantine/core';
import { IconArrowsSplit, IconRefresh, IconChartBar } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import DataInteractionPanel from 'components/DataInteractionPanel';

const ModelSelection = () => {

  return (
    <Container fluid>
      <Title order={1} id="model-selection" mt="xl" mb="md">Model Selection Techniques</Title>
      
      <Text>
        Model selection is a crucial step in the machine learning pipeline. It involves choosing the best model from a set of candidate models to solve a specific problem. The goal is to find a model that generalizes well to unseen data, avoiding both underfitting and overfitting.
      </Text>

      <Stack spacing="xl" mt="xl">
        <Section
          icon={<IconArrowsSplit size={28} />}
          title="Train-Test Split"
          id="train-test-split"
        >
          <Text>
            The train-test split is a fundamental technique in model selection. It involves dividing the dataset into two parts:
          </Text>
          <List>
            <List.Item>Training set: Used to train the model (typically 70-80% of the data)</List.Item>
            <List.Item>Test set: Used to evaluate the model's performance on unseen data (typically 20-30% of the data)</List.Item>
          </List>

          <CodeBlock
            language="python"
            code={`
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the Boston housing dataset
X, y = load_boston(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error: {rmse:.2f}")
            `}
          />
        </Section>

        <Section
          icon={<IconRefresh size={28} />}
          title="Cross-Validation Methods"
          id="cross-validation"
        >
          <Text>
            Cross-validation is a more robust method for model evaluation, especially when dealing with limited data. It provides a better estimate of the model's performance on unseen data.
          </Text>

          <Title order={3} mt="md">K-Fold Cross-Validation</Title>
          <Text>
            K-Fold Cross-Validation divides the data into K equal-sized folds. The model is trained K times, each time using K-1 folds for training and the remaining fold for validation.
          </Text>

          <Image
            src="/api/placeholder/600/300"
            alt="K-Fold Cross-Validation illustration"
            caption="K-Fold Cross-Validation process"
            mt="md"
          />

          <CodeBlock
            language="python"
            code={`
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the Boston housing dataset
X, y = load_boston(return_X_y=True)

# Create a linear regression model
model = LinearRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert MSE to RMSE
rmse_scores = np.sqrt(-scores)

print(f"Cross-validated RMSE scores: {rmse_scores}")
print(f"Mean RMSE: {np.mean(rmse_scores):.2f}")
print(f"Standard deviation of RMSE: {np.std(rmse_scores):.2f}")
            `}
          />

          <Title order={3} mt="md">Stratified K-Fold</Title>
          <Text>
            Stratified K-Fold is used for classification problems to ensure that the proportion of samples for each class is roughly the same in each fold as in the whole dataset.
          </Text>

          <Title order={3} mt="md">Time Series Cross-Validation</Title>
          <Text>
            For time series data, special cross-validation techniques are used to respect the temporal order of the data, such as TimeSeriesSplit in scikit-learn.
          </Text>

          <Title order={3} mt="md">Leave-One-Out Cross-Validation (LOOCV)</Title>
          <Text>
            LOOCV is a special case of K-Fold cross-validation where K equals the number of samples. It's computationally expensive but can be useful for small datasets.
          </Text>
        </Section>

        <Section
          icon={<IconChartBar size={28} />}
          title="Model Comparison Example"
          id="model-comparison"
        >
          <Text>
            Let's compare a Linear Regression model with a Random Forest model using cross-validation.
          </Text>

          <CodeBlock
            language="python"
            code={`
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
import numpy as np

# Load the Boston housing dataset
X, y = load_boston(return_X_y=True)

# Create models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation for both models
linear_scores = cross_val_score(linear_model, X, y, cv=5, scoring='neg_mean_squared_error')
rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert MSE to RMSE
linear_rmse = np.sqrt(-linear_scores)
rf_rmse = np.sqrt(-rf_scores)

print("Linear Regression:")
print(f"Mean RMSE: {np.mean(linear_rmse):.2f}")
print(f"Standard deviation of RMSE: {np.std(linear_rmse):.2f}")

print("\nRandom Forest:")
print(f"Mean RMSE: {np.mean(rf_rmse):.2f}")
print(f"Standard deviation of RMSE: {np.std(rf_rmse):.2f}")
            `}
          />

          <Text mt="md">
            This example demonstrates how to compare two different models using cross-validation. The model with the lower mean RMSE and lower standard deviation of RMSE is generally considered better.
          </Text>
        </Section>

        <Section
          title="Best Practices"
          id="best-practices"
        >
          <List>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Use stratified sampling</span> for classification problems to maintain class distribution across folds.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Choose an appropriate number of folds</span> (typically 5 or 10) based on dataset size and computational resources.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Use appropriate metrics</span> for your problem (e.g., RMSE for regression, AUC-ROC for binary classification).</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Be cautious of data leakage</span> - ensure that the test set remains completely unseen during the entire model selection process.</Text>
            </List.Item>
          </List>
        </Section>
      </Stack>
    </Container>
  );
};

const Section = ({ icon, title, id, children }) => (
  <Stack spacing="sm">
    <Group spacing="xs">
      {icon}
      <Title order={2} id={id}>{title}</Title>
    </Group>
    {children}
  </Stack>
);

export default ModelSelection;