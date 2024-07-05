import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import CodeBlock from "components/CodeBlock";

const ModelEvaluationValidation = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Model Evaluation and Validation</h1>
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

      <h2>Model Fitting and Inference</h2>

      <p>In supervised learning, we typically have:</p>
      <ul>
        <li>
          <InlineMath math="X \in \mathbb{R}^{n \times p}" />: Input features (n
          samples, p features)
        </li>
        <li>
          <InlineMath math="y \in \mathbb{R}^n" />: Target variable
        </li>
        <li>
          <InlineMath math="f: \mathbb{R}^p \rightarrow \mathbb{R}" />: The true
          function we're trying to approximate
        </li>
        <li>
          <InlineMath math="\hat{f}: \mathbb{R}^p \rightarrow \mathbb{R}" />:
          Our model's approximation of f
        </li>
      </ul>

      <p>
        The goal is to find <InlineMath math="\hat{f}" /> that minimizes some
        loss function <InlineMath math="L(y, \hat{f}(X))" />.
      </p>

      <h3>Model Fitting</h3>
      <p>
        Model fitting, or training, is the process of finding the best
        parameters for our model <InlineMath math="\hat{f}" /> using our
        training data.
      </p>
      <CodeBlock
        language="python"
        code={`
import numpy as np
from sklearn.linear_model import LinearRegression

# Assume X and y are our feature matrix and target vector
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model = LinearRegression()
model.fit(X, y)  # This is where the model learns from the data
        `}
      />

      <h3>Inference</h3>
      <p>
        Once the model is trained, we can use it to make predictions on new
        data:
      </p>
      <CodeBlock
        language="python"
        code={`
X_new = np.array([[3, 5], [4, 4]])
y_pred = model.predict(X_new)
print("Predictions:", y_pred)
        `}
      />

      <h2>2. Data Splitting</h2>
      <p>
        Before we can evaluate a model, we need to understand how to properly
        split our data. This is crucial for getting an unbiased estimate of our
        model's performance.
      </p>

      <h3>Training Set</h3>
      <p>
        The training set is used to train the model. It's the largest portion of
        the data, typically 60-80% of the total dataset. The model learns
        patterns from this data.
      </p>

      <h3>Validation Set</h3>
      <p>
        The validation set is used to tune hyperparameters and evaluate the
        model during the training process. It's typically 10-20% of the total
        dataset. This set helps us make choices about our model without using
        our final test set.
      </p>

      <h3>Test Set</h3>
      <p>
        The test set is used to evaluate the final model performance. It's kept
        completely separate from the training process and is typically 10-20% of
        the total dataset. This set gives us an unbiased estimate of our model's
        performance on new, unseen data.
      </p>

      <h3>Implementation</h3>
      <CodeBlock
        language="python"
        code={`
from sklearn.model_selection import train_test_split

# First split: separate test set
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: separate train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
        `}
      />

      <h2>3. Understanding Model Performance</h2>
      <p>
        Now that we understand how to split our data, let's discuss how we
        measure a model's performance.
      </p>

      <h3>Training Error</h3>
      <p>
        The training error is the error that our model makes on the training
        data. A low training error means our model fits the training data well,
        but it doesn't necessarily mean it will perform well on new data.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import mean_squared_error

y_train_pred = model.predict(X_train)
train_error = mean_squared_error(y_train, y_train_pred)
print(f"Training Error: {train_error}")
  `}
      />

      <h3>Validation Error</h3>
      <p>
        The validation error is the error on our validation set. We use this to
        tune our model and make decisions about which model or hyperparameters
        to use.
      </p>
      <CodeBlock
        language="python"
        code={`
y_val_pred = model.predict(X_val)
val_error = mean_squared_error(y_val, y_val_pred)
print(f"Validation Error: {val_error}")
  `}
      />

      <h3>Test Error</h3>
      <p>
        The test error is the error on our test set. This gives us an estimate
        of how well our model will perform on new, unseen data. It's crucial
        that we only use the test set once we've finalized our model to get an
        unbiased estimate of its performance.
      </p>
      <CodeBlock
        language="python"
        code={`
y_test_pred = model.predict(X_test)
test_error = mean_squared_error(y_test, y_test_pred)
print(f"Test Error: {test_error}")
  `}
      />

      <h3>Generalization Error</h3>
      <p>
        The generalization error is the expected error on new, unseen data. We
        estimate this using our test error, but the true generalization error
        can only be known if we had access to all possible data.
      </p>
      <p>
        In practice, we use the test error as our best estimate of the
        generalization error:
      </p>
      <CodeBlock
        language="python"
        code={`
estimated_generalization_error = test_error
print(f"Estimated Generalization Error: {estimated_generalization_error}")
  `}
      />
      <p>
        Note: The true generalization error would require evaluating the model
        on the entire population of possible data points, which is typically not
        feasible.
      </p>
      <h2>4. Overfitting and Underfitting</h2>
      <p>
        Understanding the concepts of overfitting and underfitting is crucial
        for building models that generalize well.
      </p>

      <h3>Overfitting</h3>
      <p>
        Overfitting occurs when a model learns the training data too well,
        including its noise and fluctuations. An overfit model has:
      </p>
      <ul>
        <li>Low training error, high validation/test error</li>
        <li>Complex model that captures noise in the data</li>
        <li>Poor generalization to new data</li>
      </ul>

      <h3>Underfitting</h3>
      <p>
        Underfitting occurs when a model is too simple to capture the underlying
        patterns in the data. An underfit model has:
      </p>
      <ul>
        <li>High training error, high validation/test error</li>
        <li>Overly simple model that fails to capture important patterns</li>
        <li>Poor performance on both training and new data</li>
      </ul>

      <h2>5. Bias-Variance Tradeoff</h2>
      <p>
        The expected generalization error of a model can be decomposed into
        three components:
      </p>
      <BlockMath math="E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2" />
      <p>Where:</p>
      <ul>
        <li>
          <InlineMath math="y" /> is the true value
        </li>
        <li>
          <InlineMath math="\hat{f}(x)" /> is the model's prediction
        </li>
        <li>
          <InlineMath math="\sigma^2" /> is the irreducible error
        </li>
      </ul>

      <p>
        The goal is to find the sweet spot where the combined error from bias
        and variance is minimized.
      </p>
      <h3>Bias</h3>
      <p>
        Bias is the error introduced by approximating a real-world problem,
        which may be complex, by a simplified model. High bias can lead to
        underfitting.
      </p>

      <h3>Variance</h3>
      <p>
        Variance is the error introduced by the model's sensitivity to small
        fluctuations in the training set. High variance can lead to overfitting.
      </p>

      <h2>6. Cross-Validation Techniques</h2>
      <p>
        Cross-validation is a resampling procedure used to evaluate machine
        learning models on a limited data sample. It provides a more robust
        estimate of model performance than a single train-test split.
      </p>

      <h3>K-Fold Cross-Validation</h3>
      <p>
        In K-Fold CV, the data is divided into k subsets. The model is trained
        on k-1 subsets and validated on the remaining subset. This process is
        repeated k times, with each subset serving as the validation set once.
      </p>

      <CodeBlock
        language="python"
        code={`
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV score:", scores.mean())
        `}
      />

      <h3>Stratified K-Fold Cross-Validation</h3>
      <p>
        Stratified K-Fold CV is a variation of K-Fold that ensures that the
        proportion of samples for each class is roughly the same in each fold as
        in the whole dataset. This is particularly useful for imbalanced
        datasets.
      </p>

      <h2>7. Time Series Cross-Validation</h2>
      <p>
        Time series data presents unique challenges for cross-validation due to
        its temporal nature. Traditional random splitting can lead to data
        leakage and overly optimistic performance estimates.
      </p>

      <h3>Time Series Split</h3>
      <p>
        Time Series Split is a variation of K-Fold CV that respects the temporal
        order of the data. It creates training-validation splits by
        incrementally adding samples to the training set and using the next
        chunk of data as the validation set.
      </p>

      <h3>Rolling Window Validation</h3>
      <p>
        Rolling Window Validation uses a fixed-size window that moves through
        the time series data. This approach is particularly useful when you want
        to maintain a consistent training set size and capture recent trends.
      </p>
    </Container>
  );
};

export default ModelEvaluationValidation;
