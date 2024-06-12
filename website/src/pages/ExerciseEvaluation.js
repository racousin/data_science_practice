import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const ExerciseEvaluation = () => {
  return (
    <Container>
      <h1 className="my-4">Exercise Evaluation</h1>
      <p>
        To be evaluated for the exercises in this course, you need to follow the
        following steps:
      </p>
      <Row>
        <Col>
          <h2>1. Create a GitHub account</h2>
          <p>
            If you don't already have a GitHub account, you need to create one.
            You can sign up for a free account on the GitHub website.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>2. Send an email to raphaelcousin.education@gmail.com</h2>
          <p>
            Send an email to raphaelcousin.education@gmail.com with the subject
            "2024\_github\_account\_name:$name\_surname:$surname\_username:$username",
            replacing $name, $surname, and $username with your actual name,
            surname, and GitHub username.
          </p>
          <p>
            After sending the email, you will receive an invitation to the{" "}
            <a
              href="https://github.com/racousin/data_science_practice"
              target="_blank"
              rel="noopener noreferrer"
            >
              data\_science\_practice
            </a>{" "}
            repository on GitHub.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>3. Submit your exercises</h2>
          <p>
            To submit your exercises, you need to push your solutions to the{" "}
            <code>$username/module$number/$exercises_files</code> directory in a
            specific branch.
          </p>
          <p>
            The workflow to the main branch will run tests to validate your
            answers. You can then find your score on the Student page of{" "}
            <a
              href="https://www.raphaelcousin.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              www.raphaelcousin.com
            </a>{" "}
            under your username.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default ExerciseEvaluation;
