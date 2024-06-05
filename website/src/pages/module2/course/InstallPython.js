import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const InstallPython = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Installing Python</h1>
      <p>
        In this section, you will learn how to install Python on your computer.
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <h3>Windows</h3>
          <ol>
            <li>
              Download the latest version of Python from the official website:
            </li>
            <a
              href="https://www.python.org/downloads/windows/"
              target="_blank"
              rel="noopener noreferrer"
            >
              https://www.python.org/downloads/windows/
            </a>
            <li>Run the installer and follow the prompts.</li>
            <li>
              Make sure to check the box that says "Add Python to PATH" during
              the installation process.
            </li>
          </ol>
          <h3>MacOS</h3>
          <ol>
            <li>Python is already installed on MacOS.</li>
            <li>
              To check the version of Python installed on your computer, open a
              terminal window and run the following command:
            </li>
            <CodeBlock code={`python --version`} />
          </ol>
          <h3>Linux</h3>
          <ol>
            <li>
              Python is likely already installed on your Linux distribution.
            </li>
            <li>
              To check the version of Python installed on your computer, open a
              terminal window and run the following command:
            </li>
            <CodeBlock code={`python --version`} />
            <li>
              If Python is not installed, you can install it using the package
              manager for your distribution.
            </li>
            <li>
              For example, on Ubuntu, you can install Python using the following
              command:
            </li>
            <CodeBlock code={`sudo apt-get install python3`} />
          </ol>
        </Col>
      </Row>
    </Container>
  );
};

export default InstallPython;
