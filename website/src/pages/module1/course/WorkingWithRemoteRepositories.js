import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const WorkingWithRemoteRepositories = () => {
  const commands = [
    "git remote add origin <remote_repository_url>",
    "git remote -v",
    "git fetch origin",
    "git pull origin <branch_name>",
    "git push origin <branch_name>",
    "git remote rm origin",
  ];

  return (
    <Container>
      <h2>Working with remote repositories</h2>
      <p>
        To work with remote repositories in a Git repository, you can use
        various Git commands to add, fetch, pull, and push changes to and from
        remote repositories.
      </p>
      <Row>
        <Col>
          <h3>1. Add a remote repository</h3>
          <p>Run the following command to add a remote repository:</p>
          <CodeBlock code={commands[0]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. View remote repositories</h3>
          <p>Run the following command to view remote repositories:</p>
          <CodeBlock code={commands[1]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Fetch changes from a remote repository</h3>
          <p>
            Run the following command to fetch changes from a remote repository:
          </p>
          <CodeBlock code={commands[2]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Pull changes from a remote repository</h3>
          <p>
            Run the following command to pull changes from a remote repository:
          </p>
          <CodeBlock code={commands[3]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>5. Push changes to a remote repository</h3>
          <p>
            Run the following command to push changes to a remote repository:
          </p>
          <CodeBlock code={commands[4]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>6. Remove a remote repository</h3>
          <p>Run the following command to remove a remote repository:</p>
          <CodeBlock code={commands[5]} />
        </Col>
      </Row>
    </Container>
  );
};

export default WorkingWithRemoteRepositories;
