import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const FirstStepsWithGit = () => {
  return (
    <Container fluid>
      <h2>First Steps with Git</h2>

      {/* Creating a Repository */}
      <Row>
        <Col>
          <h3 id="creating-repository">Creating a Git Repository</h3>
          <p>
            Before you can start using Git to track changes, you need to create
            a new repository. This process begins by creating a new directory
            for your project, navigating into it, and then initializing it as a
            Git repository.
          </p>
          <p>
            <strong>Step 1: Create a new directory for your project:</strong>
          </p>
          <CodeBlock code="mkdir my_project" />
          <p>
            This command creates a new folder called 'my_project' where your
            project files will reside.
          </p>

          <p>
            <strong>Step 2: Navigate into your project directory:</strong>
          </p>
          <CodeBlock code="cd my_project" />
          <p>
            This command moves the terminal's current working directory to the
            'my_project' folder.
          </p>

          <p>
            <strong>
              Step 3: Initialize the directory as a Git repository:
            </strong>
          </p>
          <CodeBlock code="git init" />
          <p>
            This command creates a new Git repository in the current directory.
            It sets up the necessary Git infrastructure within the '.git'
            directory. Here, Git will store all the metadata for the
            repository's change history.
          </p>

          <p>
            Once these steps are completed, you have successfully set up a new
            Git repository and can begin tracking changes to the files within
            this directory.
          </p>
        </Col>
      </Row>

      {/* Introduction to staging changes */}
      <Row>
        <Col>
          <h3 id="staging-changes">Staging Changes</h3>
          <p>
            Staging changes in Git involves preparing changes made to files in
            your working directory to be included in the next commit. This
            process allows you to selectively add files to your next commit
            while leaving others unchanged.
          </p>
        </Col>
      </Row>

      {/* Detailed steps */}
      <Row className="mt-4">
        <Col>
          <h4>Adding a New File and Tracking Changes</h4>
          <p>
            <strong>
              Step 1: Create a new file to add to your Git repository:
            </strong>
          </p>
          <CodeBlock code={`echo 'Initial content' > example.txt`} />
          <p>
            This command creates a new text file named 'example.txt' with some
            initial content.
          </p>

          <h4>Step 2: Check the status of your repository:</h4>
          <p>
            Use the <code>git status</code> command to see which changes are
            pending to be added to your next commit:
          </p>
          <CodeBlock code="git status" />
          <p>
            You will see 'example.txt' listed as an untracked file because Git
            has noticed a new file in the directory but it hasn't been added to
            the staging area yet.
          </p>

          <h4>Step 3: Stage the new file:</h4>
          <p>
            To add this new file to the staging area, use the{" "}
            <code>git add</code> command:
          </p>
          <CodeBlock code="git add example.txt" />
          <p>
            This command moves 'example.txt' into the staging area, making it
            ready to be included in the next commit. You can verify this by
            running <code>git status</code> again, which will now show
            'example.txt' as staged.
          </p>

          <h4>Step 4: Stage all changes in the directory:</h4>
          <p>
            If you have multiple files to stage, you can add all modified files
            to the staging area using:
          </p>
          <CodeBlock code="git add ." />
          <p>
            This command adds all new and modified files to the staging area.
            It's useful when you have several files that need to be committed
            together.
          </p>
          <h4>Step 5: Unstage a file:</h4>
          <p>
            If you decide that a file should not be included in the next commit,
            you can unstage it using the <code>git reset</code> command:
          </p>
          <CodeBlock code="git reset HEAD example.txt" />
          <p>
            This command will remove 'example.txt' from the staging area but the
            file will remain in your working directory with any changes intact.
            Running <code>git status</code> will now show the file as not staged
            for commit.
          </p>
        </Col>
      </Row>

      {/* Committing Changes */}
      <Row>
        <Col>
          <h3 id="committing-changes">What Does Committing Mean?</h3>
          <p>
            Committing in Git refers to the process of saving your staged
            changes to the local repository's history. A commit is essentially a
            snapshot of your repository at a specific point in time, allowing
            you to record the progress of your project in manageable increments.
          </p>
        </Col>
      </Row>

      {/* Detailed explanation of what happens during a commit */}
      <Row className="mt-4">
        <Col>
          <h4>What Will Be Committed?</h4>
          <p>
            Only changes that have been staged (using <code>git add</code>) will
            be included in a commit. Unstaged changes remain in your working
            directory and are not included in the commit.
          </p>
        </Col>
      </Row>

      {/* Step-by-step process to create a commit */}
      <Row className="mt-4">
        <Col>
          <h4>How to Create a Commit</h4>
          <p>
            To create a commit, you use the <code>git commit</code> command
            along with a message describing the changes made. This message is
            important for maintaining a clear, accessible history of project
            changes and should be informative and concise.
          </p>
          <CodeBlock code="git commit -m 'Your commit message'" />
          <p>
            This command captures a snapshot of the project's currently staged
            changes. The commit message should clearly describe what the changes
            do, making it easier for others (and your future self) to understand
            the purpose of the changes without needing to read the code.
          </p>
        </Col>
      </Row>

      {/* Why commits are important */}
      <Row className="mt-4">
        <Col>
          <h4>Why Committing is Important</h4>
          <p>
            Commits serve as checkpoints where individual changes can be saved
            to the project history. This allows you to revert selected changes
            if needed or compare differences over time. They are crucial for
            collaborative projects, enabling multiple developers to work on
            different features simultaneously without conflict.
          </p>
          <p>
            Committing frequently ensures that your changes are securely
            recorded in your local repository, allowing for detailed tracking of
            your project’s evolution. It also facilitates collaborative
            workflows such as branching and merging, and supports continuous
            integration practices.
          </p>
        </Col>
      </Row>

      {/* Viewing Commit History */}
      <Row>
        <Col>
          <h3 id="viewing-commit-history">Understanding Commit History</h3>
          <p>
            Commit history in Git is a record of all previous commits in the
            repository. It allows you to review changes, revert to previous
            states, and understand the chronological progression of your
            project.
          </p>
        </Col>
      </Row>

      {/* Basic command to view commit history */}
      <Row className="mt-4">
        <Col>
          <h4>Basic Command to View Commit History</h4>
          <p>
            To see a detailed list of the commit history, you can use the{" "}
            <code>git log</code> command:
          </p>
          <CodeBlock code="git log" />
          <p>
            This command displays the commit IDs, author information, dates, and
            commit messages in a detailed log format.
          </p>
        </Col>
      </Row>

      {/* More options to view commit history */}
      <Row className="mt-4">
        <Col>
          <h4>Condensed Commit History</h4>
          <p>
            For a more concise view of the commit history, use the{" "}
            <code>--oneline</code> option:
          </p>
          <CodeBlock code="git log --oneline" />
          <p>
            This option shows each commit as a single line, making it easier to
            browse through many commits quickly.
          </p>

          <h4>Visualizing Commit History as a Graph</h4>
          <p>
            To see the commit history represented as a graph, use the{" "}
            <code>--graph</code> option along with <code>--oneline</code> and{" "}
            <code>--all</code> to show all branches:
          </p>
          <CodeBlock code="git log --graph --oneline --all" />
          <p>
            This graph view provides a visual representation of branches and
            merge points in your commit history.
          </p>
        </Col>
      </Row>

      {/* Undoing Changes */}
      <Row>
        <Col>
          <h3 id="undoing-changes">Overview of Undoing Changes</h3>
          <p>
            Git provides several tools to revert or undo changes after they have
            been committed. These tools are crucial for maintaining the
            integrity and accuracy of your project history.
          </p>
        </Col>
      </Row>

      {/* Example of Committing a Change and Undoing It */}
      <Row className="mt-4">
        <Col>
          <h4>Committing and Realizing a Mistake</h4>
          <p>Imagine you've modified a file and committed the changes:</p>
          <CodeBlock code={`echo 'Modify content' > example.txt`} />
          <CodeBlock code="git add example.txt" />
          <CodeBlock code="git commit -m 'Add modify content'" />
          <p>
            After reviewing, you realize there's a mistake that needs
            correction.
          </p>
        </Col>
      </Row>
      {/* Finding a Commit Hash */}
      <Row>
        <Col>
          <h3 id="finding-commit-hash">Finding a Commit Hash</h3>
          <p>
            Before you can revert changes or checkout a previous version, you
            need to identify the commit hash. The commit hash is a unique
            identifier for each commit. You can find this by viewing the commit
            log:
          </p>
          <CodeBlock code="git log" />
          <p>
            This command displays a list of recent commits, each with a unique
            hash at the top, author information, date, and commit message. Look
            for the hash associated with the commit you are interested in
            reverting or checking out.
          </p>
        </Col>
      </Row>
      {/* Viewing Differences */}
      <Row className="mt-4">
        <Col>
          <h4>Viewing the Difference</h4>
          <p>To see what was changed with a previous commit, you can use:</p>
          <CodeBlock code="git diff <commit_hash>" />
          <p>
            This command shows the differences between the current HEAD and the
            previous commit.
          </p>
        </Col>
      </Row>

      {/* Reverting Changes */}
      <Row className="mt-4">
        <Col>
          <h4>Reverting a Commit</h4>
          <p>
            To undo the changes made by a previous commit, use the{" "}
            <code>git revert</code> command. This will create a new commit that
            undoes all changes made in the previous commit:
          </p>
          <CodeBlock code="git revert <commit_hash>" />
          <p>
            This is safe for shared branches as it does not alter commit
            history.
          </p>
        </Col>
      </Row>

      {/* Checking Out a Previous Version */}
      <Row className="mt-4">
        <Col>
          <h4>Checking Out to a Previous Version</h4>
          <p>
            If reverting is not suitable, you can checkout a previous version of
            a file or project to see or restore it as it was:
          </p>
          <CodeBlock code="git checkout <commit_hash>" />
          <p>
            This command will go back in the past. You can then return to main
            <code>HEAD</code> commit using
          </p>
          <CodeBlock code="git checkout main" />
        </Col>
      </Row>
    </Container>
  );
};

export default FirstStepsWithGit;
