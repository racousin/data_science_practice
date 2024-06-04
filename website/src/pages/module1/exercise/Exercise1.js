import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Exercise1 = () => {
  return (
    <Container>
      <h1 className="my-4">Exercise 1: Creating a File</h1>
      <p>
        In this exercise, you will create a file named "user" with your username
        inside on three lines.
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>Create a new file named "user" in your current directory.</li>
            <li>Open the file in a text editor.</li>
            <li>On the first line, write your username.</li>
            <li>Save the file.</li>
          </ol>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Example Solution</h2>
          <CodeBlock
            code={`# Open a new file named "user" in a text editor
$ touch user

# Open the file in a text editor
$ nano user

# Write your username on the first line
myusername

# Save the file and exit the text editor
# (Ctrl+X, then Y, then Enter)

# Verify that the file contains your username
$ cat user
myusername`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise1;
