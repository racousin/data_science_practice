import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const MakingChangesToFiles = () => {
  const commands = [
    "git status",
    "git diff",
    "git add <file>",
    "git commit -m 'Commit message'",
  ];

  return (
    <Container fluid>
      <h2>Making changes to files</h2>
      <p>To make changes to files in a Git repository, follow these steps:</p>
      <Row>
        <Col>
          <h3>1. Make changes to the files</h3>
          <p>
            Use a text editor to make changes to the files in your repository.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. Check the status of the repository</h3>
          <p>
            Run the following command to check the status of the repository and
            see which files have been modified:
          </p>
          <CodeBlock code={commands[0]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. View the changes</h3>
          <p>
            Run the following command to view the changes you have made to the
            files:
          </p>
          <CodeBlock code={commands[1]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Stage the changes</h3>
          <p>
            Run the following command to stage the changes you have made to a
            specific file:
          </p>
          <CodeBlock code={commands[2]} />
          <p>Repeat this step for each file you want to stage.</p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>5. Commit the changes</h3>
          <p>
            Run the following command to commit the changes you have staged,
            with a commit message that describes the changes:
          </p>
          <CodeBlock code={commands[3]} />
        </Col>
      </Row>
    </Container>
  );
};

export default MakingChangesToFiles;
