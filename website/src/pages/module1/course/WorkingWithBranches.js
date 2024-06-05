import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const WorkingWithBranches = () => {
  const commands = [
    "git branch",
    "git branch <branch_name>",
    "git checkout <branch_name>",
    "git merge <branch_name>",
    "git branch -d <branch_name>",
  ];

  return (
    <Container fluid>
      <h2>Working with branches</h2>
      <p>
        To work with branches in a Git repository, you can use various Git
        commands to create, switch, merge, and delete branches.
      </p>
      <Row>
        <Col>
          <h3>1. List all branches in the repository</h3>
          <p>
            Run the following command to list all branches in the repository:
          </p>
          <CodeBlock code={commands[0]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. Create a new branch</h3>
          <p>Run the following command to create a new branch:</p>
          <CodeBlock code={commands[1]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Switch to a different branch</h3>
          <p>Run the following command to switch to a different branch:</p>
          <CodeBlock code={commands[2]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Merge a branch into the current branch</h3>
          <p>
            Run the following command to merge a branch into the current branch:
          </p>
          <CodeBlock code={commands[3]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>5. Delete a branch</h3>
          <p>Run the following command to delete a branch:</p>
          <CodeBlock code={commands[4]} />
        </Col>
      </Row>
    </Container>
  );
};

export default WorkingWithBranches;
