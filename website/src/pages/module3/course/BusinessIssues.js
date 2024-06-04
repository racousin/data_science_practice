import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const BusinessIssues = () => {
  return (
    <Container>
      <h1 className="my-4">Business Issues</h1>
      <p>
        In this section, you will learn about the business issues that data
        science can help answer.
      </p>
      <Row>
        <Col>
          <h2>Predictive Analytics</h2>
          <p>
            Data science can be used to predict future outcomes based on
            historical data. This can help businesses make more informed
            decisions about inventory management, demand forecasting, and
            pricing strategies.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Customer Insights</h2>
          <p>
            Data science can be used to gain insights into customer behavior and
            preferences. This can help businesses improve customer satisfaction,
            loyalty, and retention.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Risk Management</h2>
          <p>
            Data science can be used to identify and mitigate risks. This can
            help businesses improve operational efficiency, reduce costs, and
            increase profitability.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default BusinessIssues;
