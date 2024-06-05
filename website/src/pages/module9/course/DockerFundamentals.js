import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DockerFundamentals = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Docker Fundamentals</h1>
      <p>In this section, you will learn the basics of Docker.</p>
      <Row>
        <Col>
          <h2>Installing and Setting up Docker</h2>
          <p>
            Docker can be installed on various operating systems, including
            Windows, macOS, and Linux. Once installed, Docker can be configured
            to run as a service or as a daemon.
          </p>
          <h2>Understanding Docker Images and Containers</h2>
          <p>
            Docker images are read-only templates that contain the instructions
            for creating a container. Containers are running instances of
            images. Images can be created manually or by using a Dockerfile,
            which is a text file that contains instructions for building an
            image.
          </p>
          <h2>Dockerfile Basics: Writing Your First Dockerfile</h2>
          <p>
            A Dockerfile is a text file that contains instructions for building
            a Docker image. It consists of a series of commands that are
            executed in order to create the image.
          </p>
          <CodeBlock
            code={`# Example of a Dockerfile
FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default DockerFundamentals;
