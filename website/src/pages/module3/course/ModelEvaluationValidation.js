import React from "react";
import { Container, Row, Col } from "react-bootstrap";
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
        </Col>
        <Col md={6}>
          <h3>Underfitting</h3>
          <ul>
            <li>Model is too simple to capture patterns</li>
            <li>High training error, high validation/test error</li>
            <li>Poor performance on all datasets</li>
          </ul>
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