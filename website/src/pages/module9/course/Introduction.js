import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Text Processing</h1>
      <p>
        In this section, you will understand the fundamentals of text processing
        within the context of data science.
      </p>
      <Row>
        <Col>
          <h2>
            Overview of Text Processing and Its Significance in Data Analytics
          </h2>
          <p>
            Text processing is the process of converting raw text data into a
            format that can be analyzed and used to extract insights. It plays a
            significant role in data analytics, enabling the analysis of
            customer reviews, sentiment analysis, spam detection, and more.
          </p>
          <h2>Challenges in Handling Text Data</h2>
          <p>Handling text data presents several challenges, including:</p>
          <ul>
            <li>Noise and inconsistencies in the data</li>
            <li>Ambiguity and subjectivity in language</li>
            <li>Large volumes of data</li>
            <li>The need for efficient and scalable solutions</li>
          </ul>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
