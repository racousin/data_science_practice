import React from "react";
import { Container, Row, Col, Card, Button } from "react-bootstrap";

const Teaching = () => {
  const modules = [
    {
      id: 1,
      title: "Module 1: Git",
      linkCourse: "/module1/course",
      linkExercise: "/module1/exercise",
    },
    {
      id: 2,
      title: "Module 2: Python environment and package",
      linkCourse: "/module2/course",
      linkExercise: "/module2/exercise",
    },
    {
      id: 3,
      title: "Module 3: Data Science Overview",
      linkCourse: "/module3/course",
      linkExercise: "/module3/exercise",
    },

    // Add more modules as needed
  ];

  return (
    <Container>
      <h1 className="my-4">Data Science in Practice</h1>
      <p>
        This course provides a practical introduction to data science. Students
        will learn the data science workflow, from data collection to model
        deployment.
      </p>
      <Row>
        {modules.map((module) => (
          <Col key={module.id} md={4} className="mb-4">
            <Card>
              <Card.Body>
                <Card.Title>{module.title}</Card.Title>
                <Button variant="primary" href={module.linkCourse}>
                  Go to Course
                </Button>
                <Button variant="primary" href={module.linkExercise}>
                  Go to Exercise
                </Button>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </Container>
  );
};

export default Teaching;
