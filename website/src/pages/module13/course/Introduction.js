import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Recommendation Systems</h1>
      <p>
        In this section, you will understand the principles and significance of
        recommendation systems.
      </p>
      <Row>
        <Col>
          <h2>Overview of Recommendation Systems</h2>
          <p>
            Recommendation systems are a type of information filtering system
            that is used to suggest items to users based on their preferences.
          </p>
          <h2>Types of Recommendation Systems</h2>
          <p>
            There are three main types of recommendation systems: content-based,
            collaborative filtering, and hybrid.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
