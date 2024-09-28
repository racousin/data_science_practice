import React from "react";
import { Container } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import CodeBlock from "components/CodeBlock";

const ModelTrainingPrediction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Model Training and Prediction</h1>

      <h2 id="model-fitting">Model Training</h2>
      <p>
        In supervised learning, we typically have:
      </p>
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

# Assume X_train and y_train are our feature matrix and target vector
model = LinearRegression()
model.fit(X_train, y_train)  # This is where the model learns from the data
        `}
      />

      <h2 id="prediction">Prediction</h2>
      <p>
        Once the model is trained, we can use it to make predictions on new data:
      </p>
      <CodeBlock
        language="python"
        code={`
# Make predictions on test set
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
        `}
      />

    </Container>
  );
};

export default ModelTrainingPrediction;