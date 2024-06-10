import React from "react";
import { Container, Row, Col, Button } from "react-bootstrap";
import { Link } from "react-router-dom";

const CreateGithubAccount = () => {
  return (
    <Container fluid>
      <h2>Create a GitHub Account</h2>
      <p>
        GitHub is a platform for hosting and collaborating on software
        development projects using Git. Creating a GitHub account is the first
        step towards managing your projects online, contributing to other
        projects, and collaborating with other developers.
      </p>
      <Row>
        <Col md={12}>
          <ol>
            <li>
              Visit the GitHub homepage:{" "}
              <a
                href="https://www.github.com"
                target="_blank"
                rel="noopener noreferrer"
              >
                www.github.com
              </a>
              .
            </li>
            <li>
              Click on the “Sign up” button in the upper-right corner of the
              homepage.
            </li>
            <li>Follow the steps!</li>
          </ol>
        </Col>
      </Row>
    </Container>
  );
};

export default CreateGithubAccount;
