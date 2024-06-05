import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const BestPracticesForUsingGit = () => {
  return (
    <Container fluid>
      <h2>Best practices for using Git</h2>
      <p>
        To use Git effectively, it's important to follow best practices such as:
      </p>
      <Row>
        <Col>
          <h3>1. Use meaningful commit messages</h3>
          <p>
            Write clear and concise commit messages that describe the changes
            you have made. This will make it easier to understand the history of
            the repository.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>2. Use branches for feature development</h3>
          <p>
            Create a new branch for each feature or bug fix you are working on.
            This will allow you to work on multiple features simultaneously and
            isolate changes to specific features.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>3. Use pull requests for code review</h3>
          <p>
            Use pull requests to review changes before they are merged into the
            main branch. This will ensure that changes are reviewed and tested
            before they are incorporated into the codebase.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>4. Keep the repository clean</h3>
          <p>
            Regularly clean up the repository by removing unnecessary files,
            merging branches that are no longer needed, and squashing commits
            that are not meaningful. This will make it easier to navigate the
            repository and understand its history.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>5. Use tags to mark important releases</h3>
          <p>
            Use tags to mark important releases of the software. This will make
            it easier to identify the version of the software that was used for
            a particular project or deployment.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default BestPracticesForUsingGit;
