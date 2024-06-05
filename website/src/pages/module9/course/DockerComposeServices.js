import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DockerComposeServices = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Docker Compose and Services</h1>
      <p>
        In this section, you will learn how to orchestrate multiple containers
        with Docker Compose.
      </p>
      <Row>
        <Col>
          <h2>Writing docker-compose.yml files</h2>
          <p>
            Docker Compose uses a YAML file to define the services and
            configurations for a multi-container application. The file is called
            `docker-compose.yml` and it contains a list of services, each with
            its own configuration.
          </p>
          <CodeBlock
            code={`# Example of a docker-compose.yml file
version: '3'
services:
  web:
    build: .
    ports:
      - "5000:5000"
  redis:
    image: "redis:alpine"`}
          />
          <h2>Managing Multi-Container Applications with Docker Compose</h2>
          <p>
            Docker Compose allows you to start, stop, and manage multiple
            containers as a single application. You can use the `docker-compose
            up` command to start all the services defined in the
            `docker-compose.yml` file.
          </p>
          <h2>Scaling and Updating Applications</h2>
          <p>
            Docker Compose allows you to scale your application by running
            multiple instances of a service. You can use the `docker-compose
            scale` command to increase or decrease the number of instances of a
            service.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default DockerComposeServices;
