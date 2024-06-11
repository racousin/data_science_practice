import React from "react";
import { Container, Row, Col, Button } from "react-bootstrap";
import { Link } from "react-router-dom";
import ModuleNavigation from "components/ModuleNavigation";

const PrerequisiteAndMethodology = () => {
  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={0}
          isCourse={null}
          title="Module 0: Prerequisites and Methodology"
        />
      </Row>
      <Row>
        <p>
          Welcome to the introductory module of the course. This section will
          guide you through the setup process and explain the methodologies we
          will use throughout the course.
        </p>
      </Row>
      <Row>
        <Col>
          <p>Last Updated: {"2024-06-07"}</p>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Setting Up Your Environment</h2>
          <p>
            To participate in the course exercises and submit your work, you
            need to set up a GitHub account and be added to the course's GitHub
            repository.
          </p>
          <ol>
            <li>
              <strong>Create a GitHub Account:</strong> If you do not have a
              GitHub account, you will need to create one. Follow the
              instructions on{" "}
              <Link to="module1/course/configure-and-access-github">
                Creating a GitHub Account
              </Link>
              .
            </li>
            <li>
              <strong>Email Registration for Course Repository:</strong> Once
              you have your GitHub username, send an email to{" "}
              <a href="mailto:raphaelcousin.education@gmail.com">
                raphaelcousin.education@gmail.com
              </a>{" "}
              with the subject
              "2024_github_account_name:$name_surname:$surname_username:$username",
              replacing $name, $surname, and $username with your actual details.
              You will receive an invitation to join the{" "}
              <i>data_science_practice</i> repository which contains the
              materials and exercises relevant to this course.
            </li>
            <li>
              <strong>Access Course Materials:</strong> After receiving the
              invitation, accept it to access the course materials and begin
              participating in the exercises.
            </li>
          </ol>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Methodology and Submitting Exercises</h2>
          <p>
            Each module in this course consists of lectures and hands-on
            exercises. Here’s how you will interact with the course content and
            submit your exercises:
          </p>
          <ul>
            <li>
              <strong>Module Structure:</strong> Each module will have a
              theoretical component accessible via GitHub and practical
              exercises that you need to complete.
            </li>
            <li>
              <strong>Submitting Exercises:</strong> For each exercise, you will
              create a new branch in the repository, add your solutions to your
              username folder under the respective module (e.g.,
              usernameFolder/module1/yourfiles), and submit a pull request.
            </li>
            <li>
              <strong>Review Process:</strong> Your submission will be reviewed
              by a collaborator (course administrator) who will provide feedback
              or approve the merge into the main branch. This process ensures
              that your work meets the required standards and does not interfere
              with the core repository structure.
            </li>
            <li>
              <strong>CI/CD Validation:</strong> Upon merging, CI/CD processes
              will run tests to validate your exercises. Successful completion
              will update your status on the course's official student results
              page.
            </li>
          </ul>
          <p>
            To see your progress and results, visit the{" "}
            <a href="https://www.raphaelcousin.com/students">
              student results page
            </a>{" "}
            on the course website.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default PrerequisiteAndMethodology;
