import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const UndoingChanges = () => {
  const commands = [
    "git checkout -- <file>",
    "git reset HEAD <file>",
    "git reset --hard <commit_hash>",
    "git revert <commit_hash>",
  ];

  return (
    <Container>
      <h2>Undoing changes</h2>
      <p>
        To undo changes in a Git repository, you can use various Git commands,
        depending on the stage of the changes.
      </p>
      <Row>
        <Col>
          <h3>1. Undo changes to a file that have not been staged</h3>
          <p>
            Run the following command to undo changes to a file that have not
            been staged:
          </p>
          <CodeBlock code={commands[0]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. Undo changes to a file that have been staged</h3>
          <p>
            Run the following command to undo changes to a file that have been
            staged:
          </p>
          <CodeBlock code={commands[1]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Undo all changes after a specific commit</h3>
          <p>
            Run the following command to undo all changes after a specific
            commit:
          </p>
          <CodeBlock code={commands[2]} />
          <p>
            This will reset the repository to the state it was in at the
            specified commit.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Undo a specific commit</h3>
          <p>Run the following command to undo a specific commit:</p>
          <CodeBlock code={commands[3]} />
          <p>
            This will create a new commit that undoes the changes made in the
            specified commit.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default UndoingChanges;
