import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const InstallingGit = () => {
  const commands = {
    mac: "brew install git",
    windows: "choco install git",
    linux: "sudo apt-get install git",
  };

  return (
    <Container fluid>
      <h2>Installing Git</h2>
      <p>
        To install Git on your computer, you can use a package manager or
        download the installer from the official Git website.
      </p>
      <Row>
        <Col>
          <h3>Mac</h3>
          <p>
            If you have Homebrew installed on your Mac, you can install Git by
            running the following command in the Terminal:
          </p>
          <CodeBlock code={commands.mac} />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>Windows</h3>
          <p>
            If you have Chocolatey installed on your Windows computer, you can
            install Git by running the following command in the Command Prompt:
          </p>
          <CodeBlock code={commands.windows} />
          <p>
            Alternatively, you can download the Git installer from the official
            Git website and run it to install Git on your computer.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>Linux</h3>
          <p>
            If you are using a Debian-based Linux distribution, you can install
            Git by running the following command in the Terminal:
          </p>
          <CodeBlock code={commands.linux} />
        </Col>
      </Row>
    </Container>
  );
};

export default InstallingGit;
