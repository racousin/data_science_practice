import React from "react";
import { Container, Row, Col, Button } from "react-bootstrap";
import { Link } from "react-router-dom";
import ModuleNavigation from "components/ModuleNavigation";
import CodeBlock from "components/CodeBlock";

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
              "2024_github_account_registration:$name:$surname:$username",
              replacing $name, $surname, and $username with your actual details.
              You will receive an invitation to join the{" "}
              <Link to="https://github.com/racousin/data_science_practice">
                <i>data_science_practice</i>
              </Link>{" "}
              repository which contains the materials and exercises relevant to
              this course.
            </li>
            <li>
              <strong>Access Course Materials:</strong> After receiving the
              invitation, accept it to access and you can start with the
              <Link to="module1/course/">Module 1: Git</Link>
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
              theoretical component accessible and practical exercises that you
              need to complete.
            </li>
            <li>
              <strong>Submitting Exercises:</strong> For each module, you will
              create a new branch in the repository, add your solutions to your
              username folder under the respective module (e.g.,
              usernameFolder/module1/yourfiles), and submit a pull request.
            </li>
            <li>
              <strong>Review Process:</strong> Your submission will be reviewed
              by a collaborator who will provide feedback or approve the merge
              into the main branch. The admin will also validate the PR before
              merging. This process ensures that your work meets the required
              standards and does not interfere with the core repository
              structure.
            </li>
            <li>
              <strong>CI/CD Validation:</strong> Upon merging to main, CI/CD
              processes will run tests to validate your exercises. Successful
              completion will update your status on the course's official
              student results page.
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
      <Row>
        <h2>Example Exercise Submission Steps</h2>
        <ol>
          <li>
            <strong>If not done, clone the repository:</strong>
            <CodeBlock
              code={`git clone https://github.com/racousin/data_science_practice.git`}
            />
          </li>
          <li>
            <strong>
              If not done, configure Git with your GitHub account:
            </strong>
            <CodeBlock code={`git config --global user.name "Your Name"`} />
            <CodeBlock
              code={`git config --global user.email "your_email@example.com"`}
            />
          </li>
          <li>
            <strong>Pull the latest main branch:</strong>
            <CodeBlock code={`git checkout main`} />
            <CodeBlock code={`git pull origin main`} />
          </li>
          <li>
            <strong>Create and checkout to a new branch:</strong>
            <CodeBlock code={`git checkout -b myusername/module1`} />
          </li>
          <li>
            <strong>Work on the exercise in your repository folder:</strong>
            <p>Example files and folder:</p>
            <ul>
              <li>myusername/module1/file.py</li>
              <li>myusername/module1/file.csv</li>
              <li>myusername/module1/my_pkg/</li>
            </ul>
          </li>
          <li>
            <strong>Stage your changes:</strong>
            <CodeBlock code={`git add myusername/module1`} />
          </li>
          <li>
            <strong>Commit your changes:</strong>
            <CodeBlock
              code={`git commit -m "My content for module 1 exercises"`}
            />
          </li>
          <li>
            <strong>Go to GitHub to create a new pull request:</strong>
            <p>
              Visit{" "}
              <a href="https://github.com/racousin/data_science_practice/pulls">
                GitHub Pull Requests
              </a>{" "}
              to create a new pull request from branch{" "}
              <code>myusername/module1</code> to <code>main</code>.
            </p>
          </li>
          <li>
            <strong>Ask for reviewers:</strong>
            <p>Add designated reviewers to the pull request for feedback.</p>
          </li>
          <li>
            <strong>
              If needed, integrate the reviewers' changes and ask for a review
              again:
            </strong>
            <p>
              Update your branch with suggested changes and re-request a review.
            </p>
          </li>
          <li>
            <strong>Merge the pull request:</strong>
            <p>Once approved, merge your pull request into the main branch.</p>
          </li>
        </ol>
      </Row>
    </Container>
  );
};

export default PrerequisiteAndMethodology;
