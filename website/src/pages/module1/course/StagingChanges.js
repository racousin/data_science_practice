import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const StagingChanges = () => {
  const commands = [
    "git status",
    "git add <file>",
    "git add .",
    "git reset <file>",
    "git reset",
  ];

  return (
    <Container>
      <h2>Staging changes</h2>
      <p>To stage changes in a Git repository, follow these steps:</p>
      <Row>
        <Col>
          <h3>1. Check the status of the repository</h3>
          <p>
            Run the following command to check the status of the repository and
            see which files have been modified:
          </p>
          <CodeBlock code={commands[0]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. Stage changes to a specific file</h3>
          <p>
            Run the following command to stage the changes you have made to a
            specific file:
          </p>
          <CodeBlock code={commands[1]} />
          <p>Repeat this step for each file you want to stage.</p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Stage all changes in the repository</h3>
          <p>
            Run the following command to stage all the changes you have made in
            the repository:
          </p>
          <CodeBlock code={commands[2]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Unstage changes to a specific file</h3>
          <p>
            Run the following command to unstage the changes you have made to a
            specific file:
          </p>
          <CodeBlock code={commands[3]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>5. Unstage all changes in the repository</h3>
          <p>
            Run the following command to unstage all the changes you have made
            in the repository:
          </p>
          <CodeBlock code={commands[4]} />
        </Col>
      </Row>
    </Container>
  );
};

export default StagingChanges;
