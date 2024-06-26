import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const BestPractices = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Best Practices and Common Pitfalls</h1>
      <p>
        In this section, you will learn about the best practices in feature
        engineering and common pitfalls to avoid.
      </p>
      <Row>
        <Col>
          <h2>Avoiding Overfitting with Feature Engineering</h2>
          <p>
            Feature engineering should be done carefully to avoid overfitting.
            This can be achieved by using techniques such as cross-validation
            and regularization.
          </p>
          <h2>
            Ensuring Model Interpretability with Sensible Feature Engineering
          </h2>
          <p>
            Feature engineering should be done in a way that makes the resulting
            model interpretable. This can be achieved by using techniques that
            are easy to understand and by avoiding complex transformations that
            may obscure the underlying data.
          </p>
          <h2>Continuous Monitoring and Updating of Features</h2>
          <p>
            Features should be continuously monitored and updated to ensure that
            they remain relevant and accurate. This can be achieved by regularly
            retraining models and updating features based on new data.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default BestPractices;
