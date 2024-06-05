import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const SettingUpPythonEnvironment = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Setting Up Python Environment</h1>
      <p>
        In this section, you will learn how to set up a Python environment using
        virtualenv.
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>Install virtualenv using pip:</li>
            <CodeBlock code={`pip install virtualenv`} />
            <li>Create a new virtual environment:</li>
            <CodeBlock code={`virtualenv myenv`} />
            <li>Activate the virtual environment:</li>
            <CodeBlock code={`source myenv/bin/activate`} />
            <li>Deactivate the virtual environment:</li>
            <CodeBlock code={`deactivate`} />
          </ol>
        </Col>
      </Row>
    </Container>
  );
};

export default SettingUpPythonEnvironment;
