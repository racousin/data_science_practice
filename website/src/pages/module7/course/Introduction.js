import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Advanced Modeling Techniques</h1>
      <p>
        In this section, you will be introduced to advanced modeling techniques
        that are particularly effective for tabular data.
      </p>
      <Row>
        <Col>
          <h2>Overview of Tabular Data Challenges</h2>
          <p>
            Tabular data is structured data that is organized in rows and
            columns. It is a common format for many real-world applications,
            such as finance, healthcare, and marketing. However, tabular data
            can present challenges for machine learning models, such as missing
            values, outliers, and high dimensionality.
          </p>
          <h2>
            Importance of Domain Knowledge in Feature Engineering for Tabular
            Models
          </h2>
          <p>
            Feature engineering is the process of creating new features from
            existing data that can improve the performance of a machine learning
            model. For tabular data, domain knowledge is particularly important
            in feature engineering. Domain knowledge can help to identify
            relevant features, create new features that capture important
            patterns in the data, and transform features in a way that makes
            sense for the problem at hand.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
