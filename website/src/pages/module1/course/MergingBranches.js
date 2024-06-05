import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const MergingBranches = () => {
  const commands = [
    "git merge <branch_name>",
    "git merge --no-ff <branch_name>",
    "git merge --abort",
  ];

  return (
    <Container fluid>
      <h2>Merging branches</h2>
      <p>
        To merge branches in a Git repository, you can use the `git merge`
        command to combine the changes from one branch into another.
      </p>
      <Row>
        <Col>
          <h3>1. Merge a branch into the current branch</h3>
          <p>
            Run the following command to merge a branch into the current branch:
          </p>
          <CodeBlock code={commands[0]} />
          <p>
            This will create a new merge commit that combines the changes from
            the specified branch into the current branch.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>
            2. Merge a branch into the current branch using a no-fast-forward
            merge
          </h3>
          <p>
            Run the following command to merge a branch into the current branch
            using a no-fast-forward merge:
          </p>
          <CodeBlock code={commands[1]} />
          <p>
            This will create a new merge commit even if the merge could be
            performed as a fast-forward merge.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Abort a merge that has conflicts</h3>
          <p>Run the following command to abort a merge that has conflicts:</p>
          <CodeBlock code={commands[2]} />
          <p>
            This will abort the merge and reset the repository to the state it
            was in before the merge was started.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default MergingBranches;
