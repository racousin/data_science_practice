import React from "react";
import { Container, Grid, Image, Title, Text, List, Box, Stack, Group, Paper, Flex } from '@mantine/core';
import { BlockMath, InlineMath } from "react-katex";
import CodeBlock from "components/CodeBlock";
const ModelEvaluation = () => {
  // todo add leaking data issue
  return (
    <Container fluid>

      <div data-slide>
        <Title order={2} id="overfitting-underfitting" mb="md">Overfitting and Underfitting</Title>
        <Grid>
          <Grid.Col span={{ md: 6 }}>
            <Title order={3} mb="sm">Overfitting</Title>
            <List spacing="sm" mb="md">
              <List.Item>Model learns noise in training data</List.Item>
              <List.Item>Low training error, high validation/test error</List.Item>
              <List.Item>Poor generalization to new data</List.Item>
            </List>
            <Image
                src="/assets/data-science-practice/module3/overfitting_illustration.png"
                alt="overfitting_illustration"
                fluid
                mb="md"
              />

          </Grid.Col>
          <Grid.Col span={{ md: 6 }}>
            <Title order={3} mb="sm">Underfitting</Title>
            <List spacing="sm" mb="md">
              <List.Item>Model is too simple to capture patterns</List.Item>
              <List.Item>High training error, high validation/test error</List.Item>
              <List.Item>Poor performance on all datasets</List.Item>
            </List>
            <Image
                src="/assets/data-science-practice/module3/underfitting_illustration.png"
                alt="underfitting_illustration"
                fluid
                mb="md"
              />

          </Grid.Col>
        </Grid>
      </div>
      <div data-slide>
        <Title order={1} mb="md">Model Evaluation</Title>
        <Text size="md" mb="md">
          Model evaluation is the process of assessing how well a machine learning
          model performs. It's a crucial step in the machine learning pipeline for
          several reasons:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>
            It helps us understand if our model is learning meaningful patterns
            from the data.
          </List.Item>
          <List.Item>
            It allows us to compare different models and choose the best one for
            our problem.
          </List.Item>
          <List.Item>
            It provides insights into how the model might perform in real-world
            scenarios.
          </List.Item>
        </List>
      </div>
      <div data-slide>
        <Title order={2} id="data-splitting" mb="md">Data Splitting</Title>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <rect x="10" y="10" width="780" height="60" rx="5" fill="#e6f3ff" stroke="#333" stroke-width="2"/>
  <text x="400" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold">Full Dataset</text>
  <rect x="10" y="100" width="624" height="60" rx="5" fill="#fff2e6" stroke="#333" stroke-width="2"/>
  <text x="322" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">Training + Validation (80%)</text>
  <rect x="644" y="100" width="146" height="60" rx="5" fill="#ffe6e6" stroke="#333" stroke-width="2"/>
  <text x="717" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">Test (20%)</text>
  <rect x="10" y="190" width="300" height="60" rx="5" fill="#e6ffe6" stroke="#333" stroke-width="2"/>
  <text x="160" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">Training Set</text>
  <rect x="320" y="190" width="304" height="60" rx="5" fill="#ffccff" stroke="#333" stroke-width="2"/>
  <text x="472" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">Validation Set</text>
  <rect x="10" y="280" width="614" height="90" rx="5" fill="#f0f0f0" stroke="#333" stroke-width="2"/>
  <text x="317" y="310" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">Model Training and Hyperparameter Optimization</text>
  <text x="317" y="340" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Train on Training Set, Validate on Validation Set</text>
  <text x="317" y="360" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Iterate to find best model and hyperparameters</text>
  <rect x="644" y="280" width="146" height="90" rx="5" fill="#ffe6e6" stroke="#333" stroke-width="2"/>
  <text x="717" y="315" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">Final Evaluation</text>
  <text x="717" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Test on</text>
  <text x="717" y="365" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Unseen Data</text>
  <line x1="400" y1="70" x2="400" y2="90" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="315" y1="160" x2="315" y2="180" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  {/* <line x1="472" y1="160" x2="472" y2="180" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/> */}
  <line x1="160" y1="250" x2="160" y2="270" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="472" y1="250" x2="472" y2="270" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="717" y1="160" x2="717" y2="270" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
</svg>
        <Text size="md" mb="md">
          To evaluate models properly, we split our data into three sets:
        </Text>
        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import train_test_split
# Split into train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split train+val into train and val
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)`}
        />
      </div>
      <div data-slide>
        <Title order={2} id="performance-metrics" mb="md">Performance Metrics</Title>
        <Text size="md" mb="md">
          We use different error metrics to assess model performance:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item><Text component="span" weight={600}>Training Error:</Text> Error on the training set</List.Item>
          <List.Item><Text component="span" weight={600}>Validation Error:</Text> Error on the validation set</List.Item>
          <List.Item><Text component="span" weight={600}>Test Error:</Text> Error on the test set (estimate of generalization error)</List.Item>
        </List>
        <CodeBlock
          language="python"
          code={`from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(y_train, model.predict(X_train))
val_error = mean_squared_error(y_val, model.predict(X_val))
test_error = mean_squared_error(y_test, model.predict(X_test))`}
        />
      </div>




      <div data-slide>
        <Title order={1} mb="lg">Hyperparameters in Machine Learning</Title>
        <Text size="lg" mb="md">
          Hyperparameters are configuration settings that control the learning process of machine learning algorithms.
          Unlike model parameters, hyperparameters are set before training begins and are not learned from data.
        </Text>
        <Text size="md" mb="md">
          Key characteristics of hyperparameters:
        </Text>
        <List spacing="sm">
          <List.Item>Set before training starts</List.Item>
          <List.Item>Control the learning algorithm behavior</List.Item>
          <List.Item>Not learned from training data</List.Item>
          <List.Item>Significantly impact model performance</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Types of Hyperparameters</Title>
        <Text size="md" mb="md">
          Hyperparameters can be categorized into several types:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item><strong>Learning Rate:</strong> Controls how much model weights are updated during training</List.Item>
          <List.Item><strong>Regularization:</strong> Prevents overfitting by adding penalties to model complexity</List.Item>
          <List.Item><strong>Model Architecture:</strong> Defines the structure of the algorithm</List.Item>
          <List.Item><strong>Training Configuration:</strong> Controls training process settings</List.Item>
        </List>
        <Text size="md">
          The choice of hyperparameters directly affects model accuracy, training time, and generalization ability.
        </Text>
      </div>
      <div data-slide>
                                    <Flex direction="column" align="center">
                              <Image
                                src="/assets/data-science-practice/module3/hyperparam.png"
                                alt="Yutong Liu & The Bigger Picture"
                                style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
                                fluid
                              />
                            </Flex>
                            </div>
      <div data-slide>
        <Title order={2} mb="md">Linear Regression Hyperparameters</Title>
        <Text size="md" mb="md">
          Linear regression hyperparameters control regularization and solver behavior:
        </Text>
        <CodeBlock
          code={`from sklearn.linear_model import LinearRegression, Ridge

# Basic Linear Regression
lr = LinearRegression(fit_intercept=True)

# Ridge Regression with regularization
ridge = Ridge(alpha=1.0, solver='auto')`}
          language="python"
        />
        <Text size="md" mt="md">
          Key hyperparameters: <InlineMath>{'\\alpha'}</InlineMath> (regularization strength),
          fit_intercept (whether to calculate intercept), solver (optimization algorithm).
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Logistic Regression Hyperparameters</Title>
        <Text size="md" mb="md">
          Logistic regression offers more hyperparameters for classification tasks:
        </Text>
        <CodeBlock
          code={`from sklearn.linear_model import LogisticRegression

# Logistic Regression with key hyperparameters
log_reg = LogisticRegression(
    C=1.0,           # Inverse regularization strength
    penalty='l2',    # Regularization type
    solver='lbfgs',  # Optimization algorithm
    max_iter=100     # Maximum iterations
)`}
          language="python"
        />
        <Text size="md" mt="md">
          C parameter: <InlineMath>{'C = \\frac{1}{\\lambda}'}</InlineMath> where <InlineMath>{'\\lambda'}</InlineMath> is regularization strength.
          Smaller C means stronger regularization.
        </Text>
      </div>

      
      <div data-slide>
        <Title order={2} id="cross-validation" mb="md">Cross-Validation</Title>
        <Text size="md" mb="md">
          Cross-validation is particularly useful when you have limited data and want to make the most of it. It provides a more reliable estimate of model performance than a single train-validation split and helps detect overfitting by testing the model's ability to generalize across different data subsets.
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
        <Group spacing="md" mb="md">
          <Group spacing="xs">
            <Box w={20} h={20} style={{ backgroundColor: '#99ccff', border: '1px solid #333' }} />
            <Text size="sm">Training Data</Text>
          </Group>
          <Group spacing="xs">
            <Box w={20} h={20} style={{ backgroundColor: '#ff9999', border: '1px solid #333' }} />
            <Text size="sm">Validation Data</Text>
          </Group>
        </Group>
        <Text size="sm" ta="center" mb="md">Each fold serves as validation data once, while the rest is used for training.</Text>
        <Text size="md" mb="md">
          Cross-validation provides a more robust estimate of model performance:
        </Text>
        <Title order={3} mb="sm">K-Fold Cross-Validation</Title>
        <List spacing="sm" mb="md" type="ordered">
          <List.Item>Split data into K folds</List.Item>
          <List.Item>Train on K-1 folds, validate on the remaining fold</List.Item>
          <List.Item>Repeat K times, using each fold as validation once</List.Item>
          <List.Item>Average the results</List.Item>
        </List>
        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print("Mean CV score:", scores.mean())`}
        />
      </div>
      <div data-slide>
        <Title order={2} id="time-series-cv" mb="md">Time Series Cross-Validation</Title>
        <Stack spacing="xs" mb="md">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <text x="400" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold">Time Series Cross-Validation</text>
  <line x1="50" y1="270" x2="750" y2="270" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <text x="750" y="290" text-anchor="end" font-family="Arial, sans-serif" font-size="14">Time</text>
  <g transform="translate(50, 60)">
    <rect x="0" y="0" width="420" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <rect x="420" y="0" width="140" height="50" fill="#ff9999" stroke="#333" stroke-width="2"/>
    <text x="-10" y="30" text-anchor="end" font-family="Arial, sans-serif" font-size="14">Fold 1</text>
    <rect x="0" y="60" width="490" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <rect x="490" y="60" width="140" height="50" fill="#ff9999" stroke="#333" stroke-width="2"/>
    <text x="-10" y="90" text-anchor="end" font-family="Arial, sans-serif" font-size="14">Fold 2</text>
    <rect x="0" y="120" width="560" height="50" fill="#99ccff" stroke="#333" stroke-width="2"/>
    <rect x="560" y="120" width="140" height="50" fill="#ff9999" stroke="#333" stroke-width="2"/>
    <text x="-10" y="150" text-anchor="end" font-family="Arial, sans-serif" font-size="14">Fold 3</text>
  </g>
  <rect x="50" y="240" width="20" height="20" fill="#99ccff" stroke="#333" stroke-width="1"/>
  <text x="80" y="255" font-family="Arial, sans-serif" font-size="14">Training Data</text>
  <rect x="200" y="240" width="20" height="20" fill="#ff9999" stroke="#333" stroke-width="1"/>
  <text x="230" y="255" font-family="Arial, sans-serif" font-size="14">Validation Data</text>
  <text x="400" y="320" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Each fold uses all available past data for training and the next time step for validation.</text>
</svg>
        </Stack>
        <Group spacing="md" mb="md">
          <Group spacing="xs">
            <Box w={20} h={20} style={{ backgroundColor: '#99ccff', border: '1px solid #333' }} />
            <Text size="sm">Training Data</Text>
          </Group>
          <Group spacing="xs">
            <Box w={20} h={20} style={{ backgroundColor: '#ff9999', border: '1px solid #333' }} />
            <Text size="sm">Validation Data</Text>
          </Group>
        </Group>
        <Text size="sm" ta="center" mb="md">Each fold uses all available past data for training and the next time step for validation.</Text>
        <Text size="md" mb="md">
          For time series data, we use specialized CV techniques:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item><Text component="span" weight={600}>Time Series Split:</Text> Respects temporal order of data</List.Item>
        </List>
        <CodeBlock
          language="python"
          code={`# Ensure your time series data is ordered
df = df.sort_values(by='date_column')
X = df[['feature_column1', 'feature_column2']].values
y = df['target_column'].values
# Sequential train-test split (70% train, 30% test)
train_size = 0.7
split_index = int(len(X) * train_size)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]`}
        />
        <CodeBlock
          language="python"
          code={`# Fit the model on (X_train, y_train)
model = LinearRegression()
model.fit(X_train, y_train)
# Test the model on (X_test, y_test)
y_pred = model.predict(X_test)
# Calculate performance metric (e.g., Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)`}
        />
        <List spacing="sm" mb="md">
          <List.Item><Text component="span" weight={600}>Rolling Window Validation:</Text> Uses fixed-size moving window</List.Item>
        </List>
        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import TimeSeriesSplit
import numpy as np
# Ensure your time series data is ordered
df = df.sort_values(by='date_column')
X = df[['feature_column1', 'feature_column2']].values
y = df['target_column'].values
tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []`}
        />
        <CodeBlock
          language="python"
          code={`# TimeSeriesSplit cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Fit the model on (X_train, y_train)
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Test the model on (X_test, y_test)
    y_pred = model.predict(X_test)
    # Calculate and store the performance metric
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    print(f"Fold {len(mse_scores)} - MSE: {mse}")
# Average MSE across all folds
average_mse = np.mean(mse_scores)
print("Average MSE across all folds:", average_mse)`}
        />
      </div>
      <div data-slide>
        <Title order={2} id="stratified-cv" mb="md">Stratified Cross-Validation</Title>
        
        <Text size="md" mb="md">
          For classification problems with imbalanced classes, stratified cross-validation ensures each fold maintains the same class distribution:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item><Text component="span" weight={600}>Preserves class ratios:</Text> Each fold has the same proportion of each class</List.Item>
          <List.Item><Text component="span" weight={600}>Reduces variance:</Text> More stable performance estimates across folds</List.Item>
          <List.Item><Text component="span" weight={600}>Better for imbalanced data:</Text> Prevents folds with too few minority class samples</List.Item>
        </List>
        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import StratifiedKFold
# For imbalanced classification data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
print("Stratified CV scores:", scores)
print("Mean CV score:", scores.mean())`}
        />
      </div>
    </Container>
  );
};
export default ModelEvaluation;