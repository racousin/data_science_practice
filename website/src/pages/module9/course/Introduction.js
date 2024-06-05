import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Containerization</h1>
      <p>
        In this section, you will learn about containerization and why it's
        useful.
      </p>
      <Row>
        <Col>
          <h2>Differences between Containers and Virtual Machines</h2>
          <p>
            Containers and virtual machines (VMs) are both technologies for
            isolating applications and their dependencies. However, there are
            some key differences between the two. Containers share the host
            system's kernel, while VMs run a full-blown operating system. This
            makes containers more lightweight and faster to start than VMs.
          </p>
          <h2>Overview of the Docker Ecosystem</h2>
          <p>
            Docker is an open-source platform for building, shipping, and
            running applications in containers. It provides a set of tools and
            services for managing containers, including Docker Engine, Docker
            Compose, and Docker Hub. Docker also has a large and active
            community of developers and users.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
