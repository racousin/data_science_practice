import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const CommittingChanges = () => {
  const commands = [
    "git commit -m 'Commit message'",
    "git commit -a -m 'Commit message'",
    "git commit --amend -m 'New commit message'",
  ];

  return (
    <Container fluid>
      <h2>Committing changes</h2>
      <p>To commit changes in a Git repository, follow these steps:</p>
      <Row>
        <Col>
          <h3>1. Stage the changes</h3>
          <p>
            Use the `git add` command to stage the changes you want to commit.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. Commit the changes</h3>
          <p>
            Run the following command to commit the changes you have staged,
            with a commit message that describes the changes:
          </p>
          <CodeBlock code={commands[0]} />
          <p>
            Alternatively, you can use the `-a` flag to stage all changes
            automatically and commit them in one step:
          </p>
          <CodeBlock code={commands[1]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Amend the last commit</h3>
          <p>
            If you want to make changes to the last commit, you can use the
            `--amend` flag to modify the commit message or add additional
            changes:
          </p>
          <CodeBlock code={commands[2]} />
          <p>
            This will replace the last commit with a new commit that includes
            the changes you have made.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default CommittingChanges;
