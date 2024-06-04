import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const InstallingPackages = () => {
  return (
    <Container>
      <h1 className="my-4">Installing Packages</h1>
      <p>In this section, you will learn how to install packages using pip.</p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>Install a package using pip:</li>
            <CodeBlock code={`pip install numpy`} />
            <li>Install a package with a specific version using pip:</li>
            <CodeBlock code={`pip install numpy==1.19.5`} />
            <li>Install packages from a requirements file using pip:</li>
            <CodeBlock code={`pip install -r requirements.txt`} />
          </ol>
        </Col>
      </Row>
    </Container>
  );
};

export default InstallingPackages;
