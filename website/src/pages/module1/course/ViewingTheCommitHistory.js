import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const ViewingTheCommitHistory = () => {
  const commands = [
    "git log",
    "git log --oneline",
    "git log --graph",
    "git log --author='Author name'",
    "git log --since='Date'",
    "git log --until='Date'",
    "git log --grep='Keyword'",
    "git show <commit_hash>",
  ];

  return (
    <Container>
      <h2>Viewing the commit history</h2>
      <p>
        To view the commit history in a Git repository, you can use the `git
        log` command with various options to filter and format the output.
      </p>
      <Row>
        <Col>
          <h3>1. View the full commit history</h3>
          <p>Run the following command to view the full commit history:</p>
          <CodeBlock code={commands[0]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. View the commit history in a condensed format</h3>
          <p>
            Run the following command to view the commit history in a condensed
            format:
          </p>
          <CodeBlock code={commands[1]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. View the commit history as a graph</h3>
          <p>
            Run the following command to view the commit history as a graph:
          </p>
          <CodeBlock code={commands[2]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Filter the commit history by author</h3>
          <p>
            Run the following command to filter the commit history by author:
          </p>
          <CodeBlock code={commands[3]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>5. Filter the commit history by date</h3>
          <p>
            Run the following commands to filter the commit history by date:
          </p>
          <CodeBlock code={commands[4]} />
          <CodeBlock code={commands[5]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>6. Filter the commit history by keyword</h3>
          <p>
            Run the following command to filter the commit history by keyword:
          </p>
          <CodeBlock code={commands[6]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>7. View the details of a specific commit</h3>
          <p>
            Run the following command to view the details of a specific commit:
          </p>
          <CodeBlock code={commands[7]} />
        </Col>
      </Row>
    </Container>
  );
};

export default ViewingTheCommitHistory;
