import React from 'react';
import { Container, Title, Text, Stack, List, Group, Image } from '@mantine/core';
import { IconArrowsSplit, IconRefresh, IconChartBar } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import DataInteractionPanel from 'components/DataInteractionPanel';


// TODO add custom metrics
// TODO add notion on time computation

const ModelSelection = () => {

  return (
    <Container fluid>
      <Title order={1} id="model-selection" mt="xl" mb="md">Model Selection</Title>
      
      <Text>
        Model selection is a crucial step in the machine learning pipeline. It involves choosing the best model from a set of candidate models to solve a specific problem. The goal is to find a model that generalizes well to unseen data, avoiding both underfitting and overfitting.
      </Text>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 280">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  {/* Models */}
  <rect x="10" y="60" width="90" height="40" fill="#ff9999" stroke="#333" stroke-width="2"/>
  <text x="55" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Model A</text>

  <rect x="10" y="110" width="90" height="40" fill="#99ccff" stroke="#333" stroke-width="2"/>
  <text x="55" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Model B</text>

  <rect x="10" y="160" width="90" height="40" fill="#90ee90" stroke="#333" stroke-width="2"/>
  <text x="55" y="185" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Model C</text>

  {/* Training Data */}
  <rect x="130" y="110" width="90" height="60" fill="#ffd700" stroke="#333" stroke-width="2"/>
  <text x="175" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Training Data</text>

  {/* Trained Models */}
  <rect x="250" y="60" width="90" height="40" fill="#ff9999" stroke="#333" stroke-width="2"/>
  <text x="295" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Trained A</text>

  <rect x="250" y="110" width="90" height="40" fill="#99ccff" stroke="#333" stroke-width="2"/>
  <text x="295" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Trained B</text>

  <rect x="250" y="160" width="90" height="40" fill="#90ee90" stroke="#333" stroke-width="2"/>
  <text x="295" y="185" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Trained C</text>

  {/* Test Data */}
  <rect x="370" y="110" width="90" height="60" fill="#ffd700" stroke="#333" stroke-width="2"/>
  <text x="415" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Test Data</text>

  {/* Scores */}
  <rect x="490" y="60" width="90" height="40" fill="#ff9999" stroke="#333" stroke-width="2"/>
  <text x="535" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Score A</text>

  <rect x="490" y="110" width="90" height="40" fill="#99ccff" stroke="#333" stroke-width="2"/>
  <text x="535" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Score B</text>

  <rect x="490" y="160" width="90" height="40" fill="#90ee90" stroke="#333" stroke-width="2"/>
  <text x="535" y="185" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Score C</text>

  {/* Best Model */}
  <rect x="610" y="110" width="90" height="60" fill="#32cd32" stroke="#333" stroke-width="2"/>
  <text x="655" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Best Model</text>

  {/* Labels */}
  <text x="63" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="16">Model Candidates</text>
  <text x="295" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="16">Trained Models</text>
  <text x="535" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="16">Model Scores</text>
  <text x="655" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="16">Selection</text>

  {/* Horizontal arrow line */}
  <line x1="10" y1="240" x2="700" y2="240" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  {/* Information text */}
  <text x="180" y="260" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">model.fit(X_tr, y_tr)</text>
  <text x="420" y="260" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">metric(model.predict(X_te), y_te)</text>
</svg>
      <Stack spacing="xl" mt="xl">
        <Title order={3} id="train-test-split">Train-Test Split</Title>
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the Boston housing dataset
X, y = load_boston(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Train a decision tree model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions on the test set for both models
y_pred_lin = lin_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

# Calculate the mean squared error and root mean squared error for both models
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)

mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)

print(f"Linear Regression RMSE: {rmse_lin:.2f}")
print(f"Decision Tree RMSE: {rmse_tree:.2f}")
# Linear Regression RMSE: 4.93
# Decision Tree RMSE: 4.24
# Best model Decision Tree: 
            `}
          />
        


    <Title order={3} id="cross-validation">Cross-Validation Methods</Title>
      <Text>
        Cross-validation is a robust method for model evaluation, especially when dealing with limited data. It provides a better estimate of the model's performance on unseen data.
      </Text>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  
  <text x="400" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold">K-Fold Cross-Validation (K=5)</text>

  <g transform="translate(50, 60)">
    <rect x="0" y="0" width="140" height="50" fill="#ff9999" stroke="#333" stroke-width="2"/>
    <rect x="140" y="0" width="560" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <text x="-10" y="30" text-anchor="end" font-family="Arial, sans-serif" font-size="14">Fold 1</text>

    <rect x="0" y="60" width="140" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <rect x="140" y="60" width="140" height="50" fill="#ff9999" stroke="#333" stroke-width="2"/>
    <rect x="280" y="60" width="420" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <text x="-10" y="90" text-anchor="end" font-family="Arial, sans-serif" font-size="14">Fold 2</text>

    <rect x="0" y="120" width="280" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <rect x="280" y="120" width="140" height="50" fill="#ff9999" stroke="#333" stroke-width="2"/>
    <rect x="420" y="120" width="280" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <text x="-10" y="150" text-anchor="end" font-family="Arial, sans-serif" font-size="14">Fold 3</text>

    <rect x="0" y="180" width="420" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <rect x="420" y="180" width="140" height="50" fill="#ff9999" stroke="#333" stroke-width="2"/>
    <rect x="560" y="180" width="140" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <text x="-10" y="210" text-anchor="end" font-family="Arial, sans-serif" font-size="14">Fold 4</text>

    <rect x="0" y="240" width="560" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <rect x="560" y="240" width="140" height="50" fill="#ff9999" stroke="#333" stroke-width="2"/>
    <text x="-10" y="270" text-anchor="end" font-family="Arial, sans-serif" font-size="14">Fold 5</text>
  </g>

  <rect x="50" y="360" width="20" height="20" fill="#99ccff" stroke="#333" stroke-width="1"/>
  <text x="80" y="375" font-family="Arial, sans-serif" font-size="14">Training Data</text>
  
  <rect x="200" y="360" width="20" height="20" fill="#ff9999" stroke="#333" stroke-width="1"/>
  <text x="230" y="375" font-family="Arial, sans-serif" font-size="14">Validation Data</text>

  <text x="400" y="395" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Each fold serves as validation data once, while the rest is used for training.</text>
</svg>
      <Title order={3} mt="md">K-Fold Cross-Validation</Title>
      <Text>
        K-Fold Cross-Validation divides the data into K equal-sized folds. The model is trained K times, each time using K-1 folds for training and the remaining fold for validation.
      </Text>

      <CodeBlock
        language="python"
        code={`
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the Boston housing dataset
X, y = load_boston(return_X_y=True)

# Initialize the model and K-fold cross-validation
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store scores
rmse_scores = []

for train_index, test_index in kf.split(X):
    # Split the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate RMSE for this fold
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    rmse_scores.append(rmse)
    
# Print summary statistics
print(f"Mean RMSE: {np.mean(rmse_scores):.2f}")
print(f"Standard deviation of RMSE: {np.std(rmse_scores):.2f}")
        `}
      />

      <Title order={3} mt="md">Stratified K-Fold</Title>
      <Text>
        Stratified K-Fold is used for classification problems to ensure that the proportion of samples for each class is roughly the same in each fold as in the whole dataset.
      </Text>

      <CodeBlock
        language="python"
        code={`from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Create a support vector classifier
svc = SVC(kernel='linear', C=1)

# Create the stratified k-fold object
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform stratified k-fold cross-validation
scores = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)

print(f"Stratified K-Fold accuracies: {scores}")
print(f"Mean accuracy: {np.mean(scores):.2f}")
print(f"Standard deviation of accuracy: {np.std(scores):.2f}")`}
      />

      <Title order={3} mt="md">Time Series Cross-Validation</Title>

      <Text>
        For time series data, special cross-validation techniques are used to respect the temporal order of the data.
      </Text>

      <CodeBlock
        language="python"
        code={`
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate sample time series data
np.random.seed(42)
X = np.array([i for i in range(100)]).reshape(-1, 1)
y = np.random.rand(100) + 0.1 * X.ravel()

# Create a linear regression model
model = LinearRegression()

# Create TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)

# Perform time series cross-validation
mse_scores = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print(f"Time Series Cross-Validation MSE scores: {mse_scores}")
print(f"Mean MSE: {np.mean(mse_scores):.2f}")
print(f"Standard deviation of MSE: {np.std(mse_scores):.2f}")
`}
      />

      <Title order={3} mt="md">Leave-One-Out Cross-Validation (LOOCV)</Title>
      <Text>
        LOOCV is a special case of K-Fold cross-validation where K equals the number of samples. It's computationally expensive but can be useful for small datasets.
      </Text>

      <CodeBlock
        language="python"
        code={`
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the Boston housing dataset
X, y = load_boston(return_X_y=True)

# Create a linear regression model
model = LinearRegression()

# Create LeaveOneOut object
loo = LeaveOneOut()

# Perform LOOCV
mse_scores = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print(f"Number of splits in LOOCV: {loo.get_n_splits(X)}")
print(f"Mean MSE: {np.mean(mse_scores):.2f}")
print(f"Standard deviation of MSE: {np.std(mse_scores):.2f}")
`}
      />
    

        <Title order={3} id="model-comparison">Model Comparison Example</Title>
          <Text>
            Let's compare a Linear Regression model with a Random Forest model using cross-validation.
          </Text>

          <CodeBlock
            language="python"
            code={`
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the Boston housing dataset
X, y = load_boston(return_X_y=True)

# Create models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store scores
linear_rmse = []
rf_rmse = []

# Perform 5-fold cross-validation for both models
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train and evaluate Linear Regression
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_rmse.append(np.sqrt(mean_squared_error(y_test, linear_pred)))
    
    # Train and evaluate Random Forest
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_rmse.append(np.sqrt(mean_squared_error(y_test, rf_pred)))

# Convert lists to numpy arrays
linear_rmse = np.array(linear_rmse)
rf_rmse = np.array(rf_rmse)

print("Linear Regression:")
print(f"Mean RMSE: {np.mean(linear_rmse):.2f}")
print(f"Standard deviation of RMSE: {np.std(linear_rmse):.2f}")
print("Random Forest:")
print(f"Mean RMSE: {np.mean(rf_rmse):.2f}")
print(f"Standard deviation of RMSE: {np.std(rf_rmse):.2f}")`}
          />

          <Text mt="md">
            This example demonstrates how to compare two different models using cross-validation. The model with the lower mean RMSE and lower standard deviation of RMSE will be considered better.
          </Text>
        

        <Title order={3} id="best-practices">Best Practices</Title>
          <List>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Choose an appropriate number of folds</span> (typically 5 or 10) based on dataset size and computational resources.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Use appropriate metrics</span> for your problem (e.g., RMSE for regression, AUC-ROC for binary classification, Custom Metric).</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Be cautious of data leakage</span> ensure that the test set remains completely unseen during the entire model selection process.</Text>
            </List.Item>
          </List>
        
      </Stack>
    </Container>
  );
};



export default ModelSelection;