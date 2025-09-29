import React from 'react';
import { Container, Title, Text, List, Alert, Flex, Image } from '@mantine/core';
import { IconClock, IconChartBar } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';

const ModelSelection = () => {

  return (
    <Container fluid>
      {/* Slide 1: Introduction */}
      <div data-slide>
        <Title order={1} id="model-selection" mt="xl" mb="md">Model Selection</Title>

        <Text mb="md">
          Model selection is a crucial step in the machine learning pipeline. It involves choosing the best model from a set of candidate models to solve a specific problem.
        </Text>

        <Text mb="md">
          <strong>Key goal:</strong> Find a model that generalizes well to unseen data, avoiding both underfitting and overfitting.
        </Text>

        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 280">
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
            </marker>
          </defs>

          {/* Models */}
          <rect x="10" y="60" width="90" height="40" fill="#ff9999" stroke="#333" strokeWidth="2"/>
          <text x="55" y="85" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Model A</text>

          <rect x="10" y="110" width="90" height="40" fill="#99ccff" stroke="#333" strokeWidth="2"/>
          <text x="55" y="135" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Model B</text>

          <rect x="10" y="160" width="90" height="40" fill="#90ee90" stroke="#333" strokeWidth="2"/>
          <text x="55" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Model C</text>

          {/* Training Data */}
          <rect x="130" y="110" width="90" height="60" fill="#ffd700" stroke="#333" strokeWidth="2"/>
          <text x="175" y="145" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Training Data</text>

          {/* Trained Models */}
          <rect x="250" y="60" width="90" height="40" fill="#ff9999" stroke="#333" strokeWidth="2"/>
          <text x="295" y="85" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Trained A</text>

          <rect x="250" y="110" width="90" height="40" fill="#99ccff" stroke="#333" strokeWidth="2"/>
          <text x="295" y="135" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Trained B</text>

          <rect x="250" y="160" width="90" height="40" fill="#90ee90" stroke="#333" strokeWidth="2"/>
          <text x="295" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Trained C</text>

          {/* Test Data */}
          <rect x="370" y="110" width="90" height="60" fill="#ffd700" stroke="#333" strokeWidth="2"/>
          <text x="415" y="145" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Test Data</text>

          {/* Scores */}
          <rect x="490" y="60" width="90" height="40" fill="#ff9999" stroke="#333" strokeWidth="2"/>
          <text x="535" y="85" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Score A</text>

          <rect x="490" y="110" width="90" height="40" fill="#99ccff" stroke="#333" strokeWidth="2"/>
          <text x="535" y="135" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Score B</text>

          <rect x="490" y="160" width="90" height="40" fill="#90ee90" stroke="#333" strokeWidth="2"/>
          <text x="535" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Score C</text>

          {/* Best Model */}
          <rect x="610" y="110" width="90" height="60" fill="#32cd32" stroke="#333" strokeWidth="2"/>
          <text x="655" y="145" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Best Model</text>

          {/* Labels */}
          <text x="63" y="40" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="16">Model Candidates</text>
          <text x="295" y="40" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="16">Trained Models</text>
          <text x="535" y="40" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="16">Model Scores</text>
          <text x="655" y="40" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="16">Selection</text>

          {/* Horizontal arrow line */}
          <line x1="10" y1="240" x2="700" y2="240" stroke="#333" strokeWidth="2" markerEnd="url(#arrowhead)"/>

          {/* Information text */}
          <text x="180" y="260" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">model.fit(X_tr, y_tr)</text>
          <text x="420" y="260" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">metric(model.predict(X_te), y_te)</text>
        </svg>
      </div>

      {/* Slide 2: Train-Test Split */}
      <div data-slide>
        <Title order={2} id="train-test-split">Train-Test Split</Title>

        <Text mb="md">
          The simplest method for model evaluation. We divide the dataset into two parts:
        </Text>

        <List mb="md">
          <List.Item>Training set: 70-80% of data for training the model</List.Item>
          <List.Item>Test set: 20-30% of data for evaluating performance</List.Item>
        </List>

        <Text mb="md">
          <strong>Step 1: Import and split the data</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

# Load data
X, y = load_boston(return_X_y=True)

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)`}
        />

        <Text mt="md" mb="md">
          <strong>Step 2: Train multiple models</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Initialize models
lin_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=42)

# Train both models
lin_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)`}
        />

        <Text mt="md" mb="md">
          <strong>Step 3: Evaluate and compare</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`import numpy as np
from sklearn.metrics import mean_squared_error

# Make predictions
y_pred_lin = lin_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

# Calculate RMSE for comparison
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))

print(f"Linear Regression RMSE: {rmse_lin:.2f}")  # 4.93
print(f"Decision Tree RMSE: {rmse_tree:.2f}")     # 4.24`}
        />

        <Text mt="md">
          Lower RMSE indicates better performance. In this case, Decision Tree performs better.
        </Text>
      </div>

      {/* Slide 3: K-Fold Cross-Validation */}
      <div data-slide>
        <Title order={2} id="cross-validation">K-Fold Cross-Validation</Title>

        <Text mb="md">
          Cross-validation provides a more robust evaluation by using multiple train-test splits. Each data point gets to be in both training and validation sets.
        </Text>

        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
          <defs>
            <marker id="arrowhead2" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
            </marker>
          </defs>

          <text x="400" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="24" fontWeight="bold">K-Fold Cross-Validation (K=5)</text>

          <g transform="translate(50, 60)">
            <rect x="0" y="0" width="140" height="50" fill="#ff9999" stroke="#333" strokeWidth="2"/>
            <rect x="140" y="0" width="560" height="50" fill="#99ccff" stroke="#333" strokeWidth="2"/>
            <text x="-10" y="30" textAnchor="end" fontFamily="Arial, sans-serif" fontSize="14">Fold 1</text>

            <rect x="0" y="60" width="140" height="50" fill="#99ccff" stroke="#333" strokeWidth="2"/>
            <rect x="140" y="60" width="140" height="50" fill="#ff9999" stroke="#333" strokeWidth="2"/>
            <rect x="280" y="60" width="420" height="50" fill="#99ccff" stroke="#333" strokeWidth="2"/>
            <text x="-10" y="90" textAnchor="end" fontFamily="Arial, sans-serif" fontSize="14">Fold 2</text>

            <rect x="0" y="120" width="280" height="50" fill="#99ccff" stroke="#333" strokeWidth="2"/>
            <rect x="280" y="120" width="140" height="50" fill="#ff9999" stroke="#333" strokeWidth="2"/>
            <rect x="420" y="120" width="280" height="50" fill="#99ccff" stroke="#333" strokeWidth="2"/>
            <text x="-10" y="150" textAnchor="end" fontFamily="Arial, sans-serif" fontSize="14">Fold 3</text>

            <rect x="0" y="180" width="420" height="50" fill="#99ccff" stroke="#333" strokeWidth="2"/>
            <rect x="420" y="180" width="140" height="50" fill="#ff9999" stroke="#333" strokeWidth="2"/>
            <rect x="560" y="180" width="140" height="50" fill="#99ccff" stroke="#333" strokeWidth="2"/>
            <text x="-10" y="210" textAnchor="end" fontFamily="Arial, sans-serif" fontSize="14">Fold 4</text>

            <rect x="0" y="240" width="560" height="50" fill="#99ccff" stroke="#333" strokeWidth="2"/>
            <rect x="560" y="240" width="140" height="50" fill="#ff9999" stroke="#333" strokeWidth="2"/>
            <text x="-10" y="270" textAnchor="end" fontFamily="Arial, sans-serif" fontSize="14">Fold 5</text>
          </g>

          <rect x="50" y="360" width="20" height="20" fill="#99ccff" stroke="#333" strokeWidth="1"/>
          <text x="80" y="375" fontFamily="Arial, sans-serif" fontSize="14">Training Data</text>

          <rect x="200" y="360" width="20" height="20" fill="#ff9999" stroke="#333" strokeWidth="1"/>
          <text x="230" y="375" fontFamily="Arial, sans-serif" fontSize="14">Validation Data</text>

          <text x="400" y="395" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Each fold serves as validation data once, while the rest is used for training.</text>
        </svg>
      </div>

      {/* Slide 4: K-Fold Implementation */}
      <div data-slide>
        <Title order={2}>K-Fold Cross-Validation Usage</Title>

        <Text mb="md">
          <strong>Step 1: Setup K-Fold</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np

# Initialize model and K-fold
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# List to store scores from each fold
rmse_scores = []`}
        />

        <Text mt="md" mb="md">
          <strong>Step 2: Train and evaluate on each fold</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`for train_idx, test_idx in kf.split(X):
    # Split data for this fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate RMSE for this fold
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    rmse_scores.append(rmse)`}
        />

        <Text mt="md" mb="md">
          <strong>Step 3: Aggregate results</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Get overall performance metrics
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print(f"Mean RMSE: {mean_rmse:.2f}")
print(f"Std Dev: {std_rmse:.2f}")  # Lower std = more consistent`}
        />

        <Text mt="md">
          The mean gives overall performance, while standard deviation shows consistency across folds.
        </Text>
      </div>

      {/* Slide 5: Stratified K-Fold */}
      <div data-slide>
        <Title order={2}>Stratified K-Fold for Classification</Title>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module6/stratified.png"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
            </Flex>
        <Text mb="md">
          For classification problems, Stratified K-Fold ensures each fold maintains the same class distribution as the original dataset.
        </Text>

        <Text mb="md">
          <strong>Why it matters:</strong> Prevents biased evaluation when classes are imbalanced.
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load classification dataset
X, y = load_iris(return_X_y=True)

# Notice: StratifiedKFold requires y for splitting
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`}
        />

        <Text mt="md" mb="md">
          <strong>Perform stratified cross-validation:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`svc = SVC(kernel='linear', C=1)
scores = []

# Note: split() needs both X and y for stratification
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    scores.append(accuracy)

print(f"Mean accuracy: {np.mean(scores):.3f}")`}
        />
      </div>

      {/* Slide 6: Time Series Cross-Validation */}
      <div data-slide>
        <Title order={2}>Time Series Cross-Validation</Title>

        <Text mb="md">
          Time series data requires special handling to respect temporal order. We can't randomly split data as future information would leak into training.
        </Text>

        <Text mb="md">
          <strong>TimeSeriesSplit:</strong> Training set always precedes the test set in time.
        </Text>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module6/tsfold.webp"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
            </Flex>
        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import TimeSeriesSplit

# Generate time series data
X = np.array(range(100)).reshape(-1, 1)
y = np.random.rand(100) + 0.1 * X.ravel()

# Create time series splitter
tscv = TimeSeriesSplit(n_splits=5)`}
        />

        <Text mt="md" mb="md">
          <strong>How splits work in time series:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {i+1}:")
    print(f"  Train: indices {train_idx[0]} to {train_idx[-1]}")
    print(f"  Test:  indices {test_idx[0]} to {test_idx[-1]}")

# Output shows expanding training window:
# Fold 1: Train 0-16, Test 17-33
# Fold 2: Train 0-33, Test 34-50
# Fold 3: Train 0-50, Test 51-66
# etc.`}
        />

        <Text mt="md">
          Each fold uses all previous data for training, ensuring no future leakage.
        </Text>
      </div>

      {/* Slide 7: Custom Metrics */}
      <div data-slide>
        <Title order={2} id="custom-metrics">Custom Metrics in Model Selection</Title>

        <Text mb="md">
          Sometimes standard metrics don't capture your specific business needs. You can create custom scoring functions for model selection.
        </Text>

        <Text mb="md">
          <strong>Example 1: Asymmetric error cost</strong>
        </Text>

        <Text mb="sm">
          When overestimation costs differ from underestimation (e.g., inventory management):
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.metrics import make_scorer

def asymmetric_mse(y_true, y_pred, over_penalty=2.0):
    """Penalize overestimation more than underestimation"""
    errors = y_pred - y_true
    return np.mean(np.where(
        errors > 0,
        over_penalty * errors**2,  # Higher penalty for overestimation
        errors**2                   # Normal penalty for underestimation
    ))`}
        />

        <Text mt="md" mb="md">
          <strong>Example 2: Business-specific metric</strong>
        </Text>

        <Text mb="sm">
          For profit optimization in pricing models:
        </Text>

        <CodeBlock
          language="python"
          code={`def profit_score(y_true, y_pred, cost_per_unit=10):
    """Calculate profit-based score"""
    revenue = y_pred * 15  # Selling price
    costs = y_pred * cost_per_unit
    lost_sales = np.maximum(0, y_true - y_pred) * 5  # Opportunity cost

    profit = revenue - costs - lost_sales
    return np.mean(profit)  # Higher is better`}
        />

        <Text mt="md" mb="md">
          <strong>Using custom metrics in cross-validation:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import cross_val_score

# Create scorer (greater_is_better=False for loss functions)
custom_scorer = make_scorer(asymmetric_mse, greater_is_better=False)

# Use in cross-validation
scores = cross_val_score(
    model, X, y, cv=5,
    scoring=custom_scorer
)
print(f"Custom metric score: {-np.mean(scores):.3f}")`}
        />
      </div>

      {/* Slide 8: More Custom Metrics Examples */}
      <div data-slide>

        <Text mt="md" mb="md">
          <strong>Multi-metric evaluation:</strong>
        </Text>

        <Text mb="sm">
          Evaluate multiple metrics simultaneously:
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import cross_validate

# Define multiple scoring metrics
scoring = {
    'mse': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'r2': 'r2',
    'custom': custom_scorer
}

# Evaluate all metrics at once
cv_results = cross_validate(
    model, X, y, cv=5,
    scoring=scoring,
    return_train_score=True
)`}
        />

        <Text mt="md" mb="md">
          <strong>Access results:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`for metric in scoring.keys():
    test_scores = cv_results[f'test_{metric}']
    print(f"{metric}: {np.mean(test_scores):.3f} (+/- {np.std(test_scores):.3f})")`}
        />
      </div>

      {/* Slide 9: Computation Time Considerations */}
      <div data-slide>
        <Title order={2} id="computation-time">
          <IconClock size={24} style={{ marginRight: 8 }} />
          Computation Time Considerations
        </Title>

        <Text mb="md">
          Model selection can be computationally expensive. Understanding time complexity helps you make informed decisions.
        </Text>

        <Text mb="md">
          <strong>Time complexity for different validation strategies:</strong>
        </Text>

        <List mb="md">
          <List.Item><strong>Train-test split:</strong> 1 × training time</List.Item>
          <List.Item><strong>K-fold CV:</strong> K × training time</List.Item>
          <List.Item><strong>LOOCV (Leave-One-Out CV):</strong> N × training time (N = number of samples)</List.Item>
        </List>

        <Text mb="md">
          <strong>Measuring computation time:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`import time
from sklearn.model_selection import KFold

def time_model_selection(model, X, y, cv_strategy):
    """Measure time for model selection"""
    start_time = time.time()

    scores = cross_val_score(model, X, y, cv=cv_strategy)

    elapsed_time = time.time() - start_time
    return scores, elapsed_time`}
        />

      </div>

      {/* Slide 10: Optimization Strategies */}
      <div data-slide>
        <Title order={2}>Optimizing Computation Time</Title>

        <Text mb="sm">
          Parallel processing : Use all available CPU cores for cross-validation:
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import cross_val_score

# n_jobs=-1 uses all available cores
scores = cross_val_score(
    model, X, y, cv=5,
    n_jobs=-1  # Parallel execution
)

# For RandomForest, also parallelize the model itself
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1, n_estimators=100)`}
        />

      </div>

      {/* Slide 11: Model Comparison */}
      <div data-slide>
        <Title order={2} id="model-comparison">Comparing Multiple Models</Title>

        <Text mb="md">
          When comparing models, use the same CV splits for fair comparison.
        </Text>

        <Text mb="md">
          <strong>Setup models to compare:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

models = {
    'Linear': LinearRegression(),
    'RF': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVM': SVR(kernel='rbf', C=1.0)
}`}
        />

        <Text mt="md" mb="md">
          <strong>Compare using same CV splits:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    results[name] = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'scores': scores
    }`}
        />

        <Text mt="md" mb="md">
          <strong>Display comparison results:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Print sorted by mean score
for name in sorted(results, key=lambda x: results[x]['mean'], reverse=True):
    r = results[name]
    print(f"{name:8} R²: {r['mean']:.3f} (+/- {r['std']:.3f})")

# Statistical significance test
from scipy import stats
t_stat, p_value = stats.ttest_rel(
    results['RF']['scores'],
    results['Linear']['scores']
)
print(f"\\nRF vs Linear p-value: {p_value:.4f}")`}
        />
      </div>

    </Container>
  );
};

export default ModelSelection;