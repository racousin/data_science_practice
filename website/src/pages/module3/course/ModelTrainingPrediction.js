import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import CodeBlock from "components/CodeBlock";

const ModelTrainingPrediction = () => {
  return (
    <Container fluid className="py-4">
      <h1 className="mb-4">Model Training and Prediction</h1>

      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>


  <rect x="10" y="10" width="180" height="80" rx="5" fill="#e6f3ff" stroke="#333" stroke-width="2"/>
  <text x="100" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Historical Data</text>
  <text x="100" y="75" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">(Features X, Labels y)</text>
  
  <line x1="190" y1="50" x2="240" y2="50" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <rect x="240" y="10" width="140" height="80" rx="5" fill="#fff2e6" stroke="#333" stroke-width="2"/>
  <text x="310" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Data Processing</text>
  
  <line x1="380" y1="50" x2="430" y2="50" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <rect x="430" y="10" width="140" height="80" rx="5" fill="#e6ffe6" stroke="#333" stroke-width="2"/>
  <text x="500" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Model Training</text>
  
  <line x1="570" y1="50" x2="620" y2="50" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <rect x="620" y="10" width="140" height="80" rx="5" fill="#ffe6e6" stroke="#333" stroke-width="2"/>
  <text x="690" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Trained Model</text>

  <rect x="10" y="250" width="180" height="80" rx="5" fill="#e6f3ff" stroke="#333" stroke-width="2"/>
  <text x="100" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">New Data</text>
  <text x="100" y="315" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">(Features X)</text>
  
  <line x1="190" y1="290" x2="240" y2="290" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <rect x="240" y="250" width="140" height="80" rx="5" fill="#fff2e6" stroke="#333" stroke-width="2"/>
  <text x="310" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Data Processing</text>
  
  <line x1="380" y1="290" x2="430" y2="290" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <rect x="430" y="250" width="140" height="80" rx="5" fill="#ffe6e6" stroke="#333" stroke-width="2"/>
  <text x="500" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Trained Model</text>
  
  <line x1="570" y1="290" x2="620" y2="290" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <rect x="620" y="250" width="140" height="80" rx="5" fill="#e6ffe6" stroke="#333" stroke-width="2"/>
  <text x="690" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Predictions</text>
  

  <path d="M690 90 L690 170 Q690 190 670 190 L480 190 Q460 190 460 210 L460 250" fill="none" stroke="#333" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrowhead)"/>
  

  <text x="400" y="140" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold">Training Process</text>
  <text x="400" y="380" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold">Serving Process</text>
</svg>

      <Row className="mb-4">
        <Col md={6}>
          <section>
            <h2 id="model-fitting">Training</h2>
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
              {/* TODO consider supervided regression, classification */}
            </ul>
            <p>
              The goal is to find <InlineMath math="\hat{f}" /> that minimizes some
              loss function <InlineMath math="L(y, \hat{f}(X))" />.
            </p>
            <p>Here's an example of model training using scikit-learn:</p>
            <CodeBlock
              language="python"
              code={`
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# 1. Data Collection (assume we have X and y)
X = np.random.rand(100, 5)
y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

# 2. Data Processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 4. Model Saving
joblib.dump(model, 'linear_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
              `}
            />
          </section>
        </Col>
        <Col md={6}>
          <section>
            <h2 id="prediction">Prediction</h2>
            <p>
              Once the model is trained, we can use it to make predictions on new data. This process involves:
            </p>
            <ol>
              <li>Loading the saved model</li>
              <li>Processing new data in the same way as the training data</li>
              <li>Using the model to generate predictions</li>
            </ol>
            <p>Here's an example of using the trained model for prediction:</p>
            <CodeBlock
              language="python"
              code={`
# 5. Model Serving
# Load the model and scaler
loaded_model = joblib.load('linear_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

# Process new data
X_new = np.random.rand(10, 5)
X_new_scaled = loaded_scaler.transform(X_new)

# Make predictions
predictions = loaded_model.predict(X_new_scaled)
print("Predictions:", predictions)
              `}
            />
          </section>
        </Col>
      </Row>

      <section className="mb-4">
        <h2 id="considerations">Key Considerations</h2>
        <ul>
          <li>Ensure consistency in data processing between training and prediction phases</li>
          <li>Regularly retrain models with new data to maintain performance</li>
          <li>Monitor model performance in production to detect drift or degradation</li>
          <li>Consider model versioning for tracking changes and facilitating rollbacks if needed</li>
        </ul>
      </section>
    </Container>
  );
};

export default ModelTrainingPrediction;