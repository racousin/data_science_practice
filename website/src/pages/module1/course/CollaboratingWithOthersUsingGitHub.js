import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const CollaboratingWithOthersUsingGitHub = () => {
  const commands = [
    "git clone <remote_repository_url>",
    "git fork <remote_repository_url>",
    "git pull-request",
    "git review",
    "git merge <pull_request_url>",
  ];

  return (
    <Container fluid>
      <h2>Collaborating with others using GitHub</h2>
      <p>
        To collaborate with others using GitHub, you can use various GitHub
        features such as forking, creating pull requests, reviewing changes, and
        merging pull requests.
      </p>
      <Row>
        <Col>
          <h3>1. Clone a remote repository</h3>
          <p>Run the following command to clone a remote repository:</p>
          <CodeBlock code={commands[0]} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. Fork a remote repository</h3>
          <p>
            Use the GitHub website to fork a remote repository. This will create
            a copy of the repository in your own GitHub account.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Create a pull request</h3>
          <p>
            Use the GitHub website to create a pull request. This allows you to
            propose changes to a remote repository and request that the changes
            be merged.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Review changes</h3>
          <p>
            Use the GitHub website to review changes proposed in a pull request.
            You can view the changes, leave comments, and approve or reject the
            changes.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>5. Merge a pull request</h3>
          <p>
            Use the GitHub website to merge a pull request. This will
            incorporate the changes proposed in the pull request into the remote
            repository.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default CollaboratingWithOthersUsingGitHub;
