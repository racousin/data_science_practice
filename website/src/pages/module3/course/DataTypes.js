import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const DataTypes = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Data Types</h1>
      <p>
        In this section, you will learn about the different types of data used
        in data science.
      </p>
      <Row>
        <Col>
          <h2>Structured Data</h2>
          <p>
            Structured data is data that has a well-defined structure, such as
            data stored in a relational database or a spreadsheet. Structured
            data is typically easier to work with and analyze than unstructured
            data.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Unstructured Data</h2>
          <p>
            Unstructured data is data that does not have a well-defined
            structure, such as text documents, images, and videos. Unstructured
            data can be more difficult to work with and analyze than structured
            data, but techniques like natural language processing and computer
            vision can be used to extract insights from unstructured data.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Semi-structured Data</h2>
          <p>
            Semi-structured data is data that has some structure, but not a
            rigid structure. Semi-structured data can be more difficult to work
            with and analyze than structured data, but techniques like XML
            parsing and JSON processing can be used to extract insights from
            semi-structured data.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default DataTypes;
