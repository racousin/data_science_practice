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
      {/* Concrete Example of Branching and Merging */}
      <Row className="mt-4">
        <Col>
          <h3 id="example-case">
            Concrete Example: Adding a Feature via Branch
          </h3>
          <p>
            Imagine you are working on a project and need to add a new feature
            without disrupting the main development line. Here’s how you can
            handle it with Git branching and merging:
          </p>
          <ol>
            <li>
              <strong>Create a Feature Branch:</strong> Suppose you want to add
              a new login feature. You would start by creating a new branch
              dedicated to this feature.
              <CodeBlock code={`git branch login-feature`} />
            </li>
            <li>
              <strong>Switch to the Feature Branch:</strong> Move to the
              'login-feature' branch to work on this feature.
              <CodeBlock code={`git checkout login-feature`} />
            </li>
            <li>
              <strong>Develop the Feature:</strong> Make all necessary changes
              for the new feature. For example, create new files or modify
              existing ones, test the feature, etc.
              <CodeBlock
                code={`git add .\ngit commit -m "Add login feature"`}
              />
            </li>
            <li>
              <strong>Switch Back to Main Branch:</strong> Once the feature
              development is complete and tested, switch back to the main branch
              to prepare for merging.
              <CodeBlock code={`git checkout main`} />
            </li>
            <li>
              <strong>Merge the Feature Branch:</strong> Merge the changes from
              'login-feature' into 'main'. Assuming no conflicts, this merge
              will integrate the new feature into the main project.
              <CodeBlock code={`git merge login-feature`} />
            </li>
            <li>
              <strong>Delete the Feature Branch:</strong> After the feature has
              been successfully merged, you can delete the branch to keep the
              repository clean.
              <CodeBlock code={`git branch -d login-feature`} />
            </li>
          </ol>
          <p>
            This workflow keeps the main line stable while allowing development
            of new features in parallel. It also ensures that any ongoing work
            is not affected by the new changes until they are fully ready to be
            integrated.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default BranchingAndMerging;
