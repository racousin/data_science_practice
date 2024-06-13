import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Exercise1 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 1: Creating and Formatting a File</h1>
      <p>
        In this exercise, you are tasked with creating a file named{" "}
        <code>user</code>
        within the <code>module1</code> directory under your username folder.
        This file should contain your username, first name, and surname,
        separated by commas.
      </p>
      <Row>
        <Col md={6}>
          <h2>Instructions</h2>
          <ol>
            <li>
              Create a new file named <code>user</code> in the directory{" "}
              <code>module1</code> under your username folder.
            </li>
            <li>Open the file in a text editor of your choice.</li>
            <li>
              Enter your username, first name, and surname in the file,
              formatted as <code>username,firstname,surname</code>.
            </li>
            <li>
              Ensure that the content is on a single line with no additional
              text or characters.
            </li>
            <li>Save and close the file.</li>
          </ol>
          <p>
            Ensure the file is saved with the correct content to pass automated
            checks.
          </p>
        </Col>
        <Col md={6}>
          <h2>Example</h2>
          <CodeBlock
            code={`// Directory Path: ./your_username/module1/user
// Content of 'user' file:
your_username,your_firstname,your_surname`}
          />
          <p>
            This block shows how your file should look when opened with a text
            editor.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise1;
