import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Feature Engineering</h1>
      <p>
        Feature engineering is the process of using domain knowledge to extract
        features (input variables) from raw data. These features can be used to
        improve the performance of machine learning models.
      </p>
      <Row>
        <Col>
          <h2>Importance and Impact of Feature Engineering</h2>
          <p>
            Feature engineering can significantly impact the performance of
            machine learning models. It can help to reduce overfitting, improve
            accuracy, and make models more interpretable.
          </p>
          <h2>Overview of the Feature Engineering Process</h2>
          <p>
            The feature engineering process typically involves data cleaning,
            data transformation, and feature selection. Data cleaning involves
            handling missing values and outliers. Data transformation involves
            scaling and normalizing data, encoding categorical data, and binning
            continuous variables. Feature selection involves selecting the most
            relevant features for the model.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
