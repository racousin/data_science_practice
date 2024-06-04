import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const ResolvingMergeConflicts = () => {
  const commands = [
    "git status",
    "git diff",
    "git add <file>",
    "git commit -m 'Resolve merge conflicts'",
  ];

  return (
    <Container>
      <h2>Resolving merge conflicts</h2>
      <p>
        To resolve merge conflicts in a Git repository, you can use a text
        editor to manually merge the conflicting changes, and then use Git to
        commit the merged changes.
      </p>
      <Row>
        <Col>
          <h3>1. Identify the conflicting files</h3>
          <p>Run the following command to identify the conflicting files:</p>
          <CodeBlock code={commands[0]} />
          <p>Git will indicate which files have merge conflicts.</p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. View the conflicting changes</h3>
          <p>Run the following command to view the conflicting changes:</p>
          <CodeBlock code={commands[1]} />
          <p>Git will mark the conflicting changes with conflict markers.</p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Resolve the conflicts</h3>
          <p>
            Use a text editor to manually merge the conflicting changes. Remove
            the conflict markers and make the necessary changes to resolve the
            conflicts.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Commit the merged changes</h3>
          <p>
            Run the following commands to stage the merged changes and commit
            them:
          </p>
          <CodeBlock code={commands[2]} />
          <CodeBlock code={commands[3]} />
          <p>
            This will create a new commit that resolves the merge conflicts.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default ResolvingMergeConflicts;
