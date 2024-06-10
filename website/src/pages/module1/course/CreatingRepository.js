import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const CreatingRepository = () => {
  const commands = {
    createLocalRepo: "git init",
    stageFiles: "git add .",
    initialCommit: "git commit -m 'Initial commit'",
    cloneRepo: "git clone https://github.com/your-username/your-repository.git",
    addRemote:
      "git remote add origin https://github.com/your-username/your-repository.git",
    pushChanges: "git push -u origin main",
  };

  return (
    <Container fluid>
      <h2>Creating and Managing a Git Repository</h2>
      <p>
        This guide covers how to create a new repository on GitHub, clone it to
        your local computer, and push changes back to GitHub.
      </p>
      <h3>Creating a Repository on GitHub</h3>
      <ol>
        <li>Log in to your GitHub account.</li>
        <li>Navigate to the Repositories tab and click 'New'.</li>
        <li>
          Enter a name for your repository and select the visibility (public or
          private).
        </li>
        <li>
          Optionally, initialize the repository with a README, .gitignore, or
          license.
        </li>
        <li>Click 'Create repository'.</li>
      </ol>

      <h3>Cloning the Repository</h3>
      <p>
        After creating your GitHub repository, you can clone it to your local
        computer to start working on the project.
      </p>
      <CodeBlock code={commands.cloneRepo} />

      <h3>Adding a New File and Committing Changes</h3>
      <p>
        Navigate into your cloned repository directory, create or modify files,
        then run the following commands to commit changes:
      </p>
      <CodeBlock code={commands.stageFiles} />
      <p>This will stage all changes made in the repository for commit.</p>
      <CodeBlock code={commands.initialCommit} />
      <p>This will commit your staged files with the initial commit message.</p>

      <h3>Pushing Changes to GitHub</h3>
      <p>
        After committing your changes locally, you can push them back to GitHub
        to make them available to others.
      </p>
      <CodeBlock code={commands.pushChanges} />

      <p>
        Following these steps, you will have a fully functional Git repository
        both locally and on GitHub, ready for further development and
        collaboration.
      </p>

      <h3>Viewing Your Changes on GitHub</h3>
      <p>
        Once you have pushed your changes to GitHub, you can view them directly
        on the GitHub website:
      </p>
      <ol>
        <li>Go to the GitHub website and navigate to your repository.</li>
        <li>
          Click on the 'Commits' link to view the list of commits. You will see
          your most recent commit at the top of the list.
        </li>
        <li>
          Click on your latest commit to view the detailed changes or browse the
          'Code' tab to see the current state of your repository.
        </li>
      </ol>
      <p>
        This allows you to track your progress, review changes, and manage your
        project directly from the web interface.
      </p>
    </Container>
  );
};

export default CreatingRepository;
