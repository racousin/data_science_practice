import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Model Building</h1>
      <p>
        Model building is the process of creating a mathematical representation
        of a real-world problem. In machine learning, models are used to make
        predictions or decisions based on data.
      </p>
      <Row>
        <Col>
          <h2>Overview of Model Building</h2>
          <p>
            Model building involves several steps, including data preprocessing,
            feature engineering, model selection, model training, and model
            evaluation.
          </p>
          <h2>Types of Machine Learning Models</h2>
          <p>
            Machine learning models can be broadly categorized into three types:
            supervised learning, unsupervised learning, and reinforcement
            learning. Supervised learning models are trained on labeled data and
            are used for tasks such as classification and regression.
            Unsupervised learning models are trained on unlabeled data and are
            used for tasks such as clustering and dimensionality reduction.
            Reinforcement learning models are trained through trial and error
            and are used for tasks such as game playing and robotics.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
