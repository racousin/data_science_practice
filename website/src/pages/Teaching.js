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
    {
      id: 4,
      title: "Module 4: Get Data",
      linkCourse: "/module4/course",
      linkExercise: "/module4/exercise",
    },
    {
      id: 5,
      title: "Module 5: Feature Engineering",
      linkCourse: "/module5/course",
      linkExercise: "/module5/exercise",
    },
    {
      id: 6,
      title: "Module 6: Model Building And Evaluation",
      linkCourse: "/module6/course",
      linkExercise: "/module6/exercise",
    },
    {
      id: 7,
      title: "Module 7: AdvancedTabularModels",
      linkCourse: "/module7/course",
      linkExercise: "/module7/exercise",
    },
    {
      id: 8,
      title: "Module 8: Deep Learning Fundamentals",
      linkCourse: "/module8/course",
      linkExercise: "/module8/exercise",
    },
    {
      id: 9,
      title: "Module 9: Docker",
      linkCourse: "/module9/course",
      linkExercise: "/module9/exercise",
    },
    {
      id: 10,
      title: "Module 10: Cloud Integration",
      linkCourse: "/module10/course",
      linkExercise: "/module10/exercise",
    },
    {
      id: 11,
      title: "Module 11: Image processing",
      linkCourse: "/module11/course",
      linkExercise: "/module11/exercise",
    },
    {
      id: 12,
      title: "Module 12: Text Processing",
      linkCourse: "/module12/course",
      linkExercise: "/module12/exercise",
    },
    {
      id: 13,
      title: "Module 13: Recommendation Systems",
      linkCourse: "/module13/course",
      linkExercise: "/module13/exercise",
    },
    {
      id: 14,
      title: "Module 14: Generative Models",
      linkCourse: "/module14/course",
      linkExercise: "/module14/exercise",
    },
    {
      id: 15,
      title: "Module 15: Reinforcement Learning",
      linkCourse: "/module15/course",
      linkExercise: "/module15/exercise",
    },
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
            <Card className="h-100">
              <Card.Body>
                <Card.Title>{module.title}</Card.Title>
                <div>
                  <Button
                    variant="outline-primary"
                    href={module.linkCourse}
                    className="button"
                  >
                    Go to Course
                  </Button>
                  <Button
                    variant="outline-secondary"
                    href={module.linkExercise}
                    className="button"
                  >
                    Go to Exercise
                  </Button>
                </div>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </Container>
  );
};

export default Teaching;
