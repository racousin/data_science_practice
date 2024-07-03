import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const WorkingWithContainers = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Working with Docker Containers</h1>
      <p>In this section, you will learn how to manage Docker containers.</p>
      <Row>
        <Col>
          <h2>Running Containers, Managing Container Lifecycles</h2>
          <p>
            Containers can be started, stopped, and restarted using the Docker
            CLI. Containers can be run in the foreground or in the background.
            When a container is started, it is assigned a unique ID that can be
            used to manage its lifecycle.
          </p>
          <CodeBlock
            code={`# Example of running a container in the background
docker run -d -p 8000:8000 my-image`}
          />
          <h2>Networking Between Containers</h2>
          <p>
            Docker provides a virtual network for containers to communicate with
            each other. By default, containers can communicate with each other
            on the same network. However, it is possible to create custom
            networks and configure network isolation.
          </p>
          <h2>Persistent Data and Volumes</h2>
          <p>
            Containers are ephemeral, which means that any data stored inside a
            container is lost when the container is stopped or deleted. To
            persist data, Docker provides volumes, which are directories that
            are managed by Docker and can be mounted into containers.
          </p>
          <CodeBlock
            code={`# Example of creating a volume and mounting it into a container
docker volume create my-volume
docker run -d -p 8000:8000 -v my-volume:/app/data my-image`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default WorkingWithContainers;
