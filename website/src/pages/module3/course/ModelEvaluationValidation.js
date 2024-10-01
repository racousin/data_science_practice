import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import CodeBlock from "components/CodeBlock";

const ModelEvaluation = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Model Evaluation</h1>

      <p>
        Model evaluation is the process of assessing how well a machine learning
        model performs. It's a crucial step in the machine learning pipeline for
        several reasons:
      </p>
      <ul>
        <li>
          It helps us understand if our model is learning meaningful patterns
          from the data.
        </li>
        <li>
          It allows us to compare different models and choose the best one for
          our problem.
        </li>
        <li>
          It provides insights into how the model might perform in real-world
          scenarios.
        </li>
      </ul>

      <h2 id="data-splitting">Data Splitting</h2>

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

      <p>
        To evaluate models properly, we split our data into three sets:
      </p>
      <BlockMath math="D = D_{train} \cup D_{val} \cup D_{test}" />
      <ul>
        <li><strong>Training set (D_train):</strong> Used to fit the model (60-80% of data)</li>
        <li><strong>Validation set (D_val):</strong> Used for tuning and model selection (10-20% of data)</li>
        <li><strong>Test set (D_test):</strong> Used for final performance estimation (10-20% of data)</li>
      </ul>
      <CodeBlock
        language="python"
        code={`
from sklearn.model_selection import train_test_split

# Split into train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split train+val into train and val
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        `}
      />

      <h2 id="performance-metrics">Performance Metrics</h2>
      <p>
        We use different error metrics to assess model performance:
      </p>
      <ul>
        <li><strong>Training Error:</strong> Error on the training set</li>
        <li><strong>Validation Error:</strong> Error on the validation set</li>
        <li><strong>Test Error:</strong> Error on the test set (estimate of generalization error)</li>
      </ul>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import mean_squared_error

train_error = mean_squared_error(y_train, model.predict(X_train))
val_error = mean_squared_error(y_val, model.predict(X_val))
test_error = mean_squared_error(y_test, model.predict(X_test))
        `}
      />

      <h2 id="overfitting-underfitting">Overfitting and Underfitting</h2>
      <Row>
        <Col md={6}>
          <h3>Overfitting</h3>
          <ul>
            <li>Model learns noise in training data</li>
            <li>Low training error, high validation/test error</li>
            <li>Poor generalization to new data</li>
          </ul>
          <Image
              src="/assets/module3/overfitting_illustration.png"
              alt="overfitting_illustration"
              fluid
            />
                  <CodeBlock
        language="python"
        code={`
from numpy.polynomial import Polynomial

# Generate training data (complete noise between 0 and 5)
x_train = np.linspace(0, 5, 10)
y_train = np.random.rand(10)

# Generate testing data (complete noise between 5 and 10)
x_test = np.linspace(5, 10, 50)
y_test = np.random.rand(50)

# Fit a high-degree polynomial (overfitting)
poly = Polynomial.fit(x_train, y_train, deg=12)

# Calculate predictions
y_train_pred = poly(x_train)
y_test_pred = poly(x_test)

# Calculate errors
train_error = np.mean((y_train - y_train_pred)**2)
test_error = np.mean((y_test - y_test_pred)**2)
        `}
      />
        </Col>
        <Col md={6}>
          <h3>Underfitting</h3>
          <ul>
            <li>Model is too simple to capture patterns</li>
            <li>High training error, high validation/test error</li>
            <li>Poor performance on all datasets</li>
          </ul>
          <Image
              src="/assets/module3/underfitting_illustration.png"
              alt="underfitting_illustration"
              fluid
            />
                  <CodeBlock
        language="python"
        code={`
from sklearn.linear_model import LinearRegression

# Generate data
x = np.linspace(0, 99, 100)
y = x % 2 + np.random.normal(0, 0.01, 100)

# Split into training and testing sets
x_train, y_train = x[:50], y[:50]
x_test, y_test = x[50:], y[50:]

# Fit a linear model (underfitting)
model = LinearRegression()
model.fit(x_train.reshape(-1, 1), y_train)

# Calculate predictions
y_train_pred = model.predict(x_train.reshape(-1, 1))
y_test_pred = model.predict(x_test.reshape(-1, 1))

# Calculate errors
train_error = np.mean((y_train - y_train_pred)**2)
test_error = np.mean((y_test - y_test_pred)**2)
        `}
      />       
        </Col>
      </Row>

      <h2 id="bias-variance">Bias-Variance Tradeoff</h2>
      <p>
        The generalization error can be decomposed into:
      </p>
      <BlockMath math="E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2" />
      <ul>
        <li><strong>Bias:</strong> Error from oversimplifying the model</li>
        <li><strong>Variance:</strong> Error from model's sensitivity to training data</li>
        <li><strong>Irreducible Error (σ²):</strong> Inherent noise in the problem</li>
      </ul>

      <h2 id="cross-validation">Cross-Validation</h2>
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
      <p>
        Cross-validation provides a more robust estimate of model performance:
      </p>
      <h3>K-Fold Cross-Validation</h3>
      <ol>
        <li>Split data into K folds</li>
        <li>Train on K-1 folds, validate on the remaining fold</li>
        <li>Repeat K times, using each fold as validation once</li>
        <li>Average the results</li>
      </ol>
      <CodeBlock
        language="python"
        code={`
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Mean CV score:", scores.mean())
        `}
      />

      <h2 id="time-series-cv">Time Series Cross-Validation</h2>
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
      <p>
        For time series data, we use specialized CV techniques:
      </p>
      <ul>
        <li><strong>Time Series Split:</strong> Respects temporal order of data</li>
        <li><strong>Rolling Window Validation:</strong> Uses fixed-size moving window</li>
      </ul>
    </Container>
  );
};

export default ModelEvaluation;