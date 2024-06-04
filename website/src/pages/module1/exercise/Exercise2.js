import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Exercise2 = () => {
  return (
    <Container>
      <h1 className="my-4">Exercise 2: Making Changes to a File</h1>
      <p>
        In this exercise, you will make changes to the "user" file created in
        Exercise 1.
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>Open the "user" file in a text editor.</li>
            <li>On the second line, write your first name.</li>
            <li>On the third line, write your last name.</li>
            <li>Save the file.</li>
          </ol>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Example Solution</h2>
          <CodeBlock
            code={`# Open the "user" file in a text editor
$ nano user

# Write your first name on the second line
John

# Write your last name on the third line
Doe

# Save the file and exit the text editor
# (Ctrl+X, then Y, then Enter)

# Verify that the file contains your first and last names
$ cat user
myusername
John
Doe`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise2;
