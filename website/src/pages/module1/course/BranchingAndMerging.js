import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const BranchingAndMerging = () => {
  return (
    <Container fluid>
      <h2>Branching and Merging</h2>
      <p>
        Branching and merging are vital features of Git that facilitate
        simultaneous and non-linear development among teams. Branching allows
        multiple developers to work on different features at the same time
        without interfering with each other, while merging brings those changes
        together into a single branch.
      </p>

      {/* Working with Branches */}
      <Row>
        <Col>
          <h3 id="working-with-branches">Working with Branches</h3>
          <p>
            Branches in Git are incredibly lightweight, making branching and
            switching between branches quick and easy:
          </p>
          <ol>
            <li>
              <strong>Create a Branch:</strong> Use{" "}
              <code>git branch new-branch-name</code> to create a new branch.
            </li>
            <li>
              <strong>Switch to a Branch:</strong> Use{" "}
              <code>git checkout new-branch-name</code> to switch to your new
              branch and start working independently from other branches.
            </li>
            <li>
              <strong>List Branches:</strong> Use <code>git branch</code> to
              list all local branches. Add <code>-a</code> to see remote
              branches as well.
            </li>
          </ol>
          <CodeBlock
            code={`git branch\n git checkout new-branch-name\n git branch -a`}
          />
        </Col>
      </Row>

      {/* Merging Branches */}
      <Row className="mt-4">
        <Col>
          <h3 id="merging-branches">Merging Branches</h3>
          <p>
            Once development on a branch is complete, the changes can be merged
            back into the main branch (e.g., 'main' or 'master'):
          </p>
          <ul>
            <li>
              <strong>Standard Merge:</strong> Use{" "}
              <code>git merge branch-name</code> from the receiving branch to
              integrate changes.
            </li>
            <li>
              <strong>No-Fast-Forward Merge:</strong> Use{" "}
              <code>git merge --no-ff branch-name</code> to ensure a new commit
              is made even if the merge could be performed with a fast-forward.
            </li>
          </ul>
          <CodeBlock
            code={`git checkout main\n git merge new-branch-name\n git merge --no-ff new-branch-name`}
          />
        </Col>
      </Row>

      {/* Resolving Merge Conflicts */}
      <Row className="mt-4">
        <Col>
          <h3 id="resolving-merge-conflicts">Resolving Merge Conflicts</h3>
          <p>
            Conflicts occur when the same parts of the same file are changed in
            different branches:
          </p>
          <ol>
            <li>
              <strong>Identify Conflicts:</strong> During a merge, Git will tell
              you if there are conflicts that need manual resolution.
            </li>
            <li>
              <strong>Edit Files:</strong> Open the conflicted files and make
              the necessary changes to resolve conflicts.
            </li>
            <li>
              <strong>Mark as Resolved:</strong> Use <code>git add</code> on the
              resolved files to mark them as resolved.
            </li>
            <li>
              <strong>Complete the Merge:</strong> Use <code>git commit</code>{" "}
              to complete the merge.
            </li>
          </ol>
          <CodeBlock
            code={`git add resolved-file.txt\n git commit -m "Resolved merge conflict by including both suggestions."`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default BranchingAndMerging;
