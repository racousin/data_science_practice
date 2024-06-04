import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const CreatingRepository = () => {
  const commands = ["git init", "git add .", "git commit -m 'Initial commit'"];

  return (
    <Container>
      <h2>Creating a Git repository</h2>
      <p>
        To create a new Git repository, navigate to the directory where you want
        to create the repository and run the following commands:
      </p>
      <Row>
        <Col>
          <CodeBlock code={commands[0]} />
          <p>
            This will initialize a new Git repository in the current directory.
          </p>
          <CodeBlock code={commands[1]} />
          <p>This will stage all the files in the repository for commit.</p>
          <CodeBlock code={commands[2]} />
          <p>
            This will create a new commit with the message "Initial commit".
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default CreatingRepository;
