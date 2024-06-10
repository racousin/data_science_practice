import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const CommittingChanges = () => {
  const commands = {
    add: "git add <file_or_directory>",
    commit: "git commit -m 'Commit message'",
    commitAll: "git commit -a -m 'Commit message'",
    amend: "git commit --amend -m 'New commit message'",
    viewCommits: "git log",
    viewDiff: "git diff HEAD~1..HEAD",
  };

  return (
    <Container fluid>
      <h2>Understanding and Committing Changes in Git</h2>
      <p>
        This guide will walk you through how to commit changes in Git, providing
        a clear history and trackable progress of your project.
      </p>
      <Row>
        <Col md={12}>
          <h3>1. Stage Changes</h3>
          <p>
            Before committing, you must stage changes that you want to include
            in your commit. This step allows you to review and select changes
            that should be part of the next snapshot.
          </p>
          <CodeBlock code={commands.add} />
          <p>
            Use the above command to add specific files or directories, or use{" "}
            <code>git add .</code> to add all modified files.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col md={12}>
          <h3>2. Commit Staged Changes</h3>
          <p>
            Once staging is complete, you can commit your changes. A commit is a
            snapshot of your repository at a specific point in time.
          </p>
          <CodeBlock code={commands.commit} />
          <p>
            If you want to skip the staging area and commit all changes that
            have been made since the last commit, use the following command:
          </p>
          <CodeBlock code={commands.commitAll} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col md={12}>
          <h3>3. Amend the Last Commit</h3>
          <p>
            If you need to correct the last commit—for example, if you forgot to
            include a file or made a typo in the commit message—you can amend
            it.
          </p>
          <CodeBlock code={commands.amend} />
          <p>
            Note that amending changes the commit history and should be used
            cautiously, especially if the commit has been shared with others.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col md={12}>
          <h3>4. View Commit History</h3>
          <p>
            To see a log of all commits, use the following command, which
            provides a list of commits, each with a unique SHA, author
            information, date, and message:
          </p>
          <CodeBlock code={commands.viewCommits} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col md={12}>
          <h3>5. View Changes Between Commits</h3>
          <p>
            To see what has changed from one commit to another, you can compare
            two commits. This is particularly useful for understanding what has
            changed in the most recent commit:
          </p>
          <CodeBlock code={commands.viewDiff} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col md={12}>
          <h3>6. Pushing Changes to a Remote Repository</h3>
          <p>
            After committing your changes locally, you should push them to a
            remote repository to make them available to others and ensure all
            team members have the latest version of the project.
          </p>
          <CodeBlock code="git push" />
          <p>
            This command transmits your local branch commits to the remote
            repository branch.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default CommittingChanges;

import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const MakingChangesToFiles = () => {
  const commands = [
    "git status",
    "git diff",
    "git add <file>",
    "git commit -m 'Commit message'",
  ];

  return (
    <Container fluid>
      <h2>Making changes to files</h2>
      <p>To make changes to files in a Git repository, follow these steps:</p>
      <Row>
        <Col>
          <h3>1. Make changes to the files</h3>
          <p>
            Use a text editor to make changes to the files in your repository.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. Check the status of the repository</h3>
          <p>
            Run the following command to check the status of the repository and
            see which files have been modified:
          </p>
          <CodeBlock code={commands[0]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. View the changes</h3>
          <p>
            Run the following command to view the changes you have made to the
            files:
          </p>
          <CodeBlock code={commands[1]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Stage the changes</h3>
          <p>
            Run the following command to stage the changes you have made to a
            specific file:
          </p>
          <CodeBlock code={commands[2]} />
          <p>Repeat this step for each file you want to stage.</p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>5. Commit the changes</h3>
          <p>
            Run the following command to commit the changes you have staged,
            with a commit message that describes the changes:
          </p>
          <CodeBlock code={commands[3]} />
        </Col>
      </Row>
    </Container>
  );
};

export default MakingChangesToFiles;

import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const StagingChanges = () => {
  const commands = [
    "git status",
    "git add <file>",
    "git add .",
    "git reset <file>",
    "git reset",
  ];

  return (
    <Container fluid>
      <h2>Staging changes</h2>
      <p>To stage changes in a Git repository, follow these steps:</p>
      <Row>
        <Col>
          <h3>1. Check the status of the repository</h3>
          <p>
            Run the following command to check the status of the repository and
            see which files have been modified:
          </p>
          <CodeBlock code={commands[0]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. Stage changes to a specific file</h3>
          <p>
            Run the following command to stage the changes you have made to a
            specific file:
          </p>
          <CodeBlock code={commands[1]} />
          <p>Repeat this step for each file you want to stage.</p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Stage all changes in the repository</h3>
          <p>
            Run the following command to stage all the changes you have made in
            the repository:
          </p>
          <CodeBlock code={commands[2]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Unstage changes to a specific file</h3>
          <p>
            Run the following command to unstage the changes you have made to a
            specific file:
          </p>
          <CodeBlock code={commands[3]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>5. Unstage all changes in the repository</h3>
          <p>
            Run the following command to unstage all the changes you have made
            in the repository:
          </p>
          <CodeBlock code={commands[4]} />
        </Col>
      </Row>
    </Container>
  );
};

export default StagingChanges;

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
    <Container fluid>
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
    <Container fluid>
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
