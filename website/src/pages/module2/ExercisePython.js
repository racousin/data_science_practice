import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const ExercisePython = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1",
      component: lazy(() => import("pages/module2/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module2/exercise/Exercise2")),
    },
    // Add links to other exercises as needed
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={2}
          isCourse={false}
          title="Module 2: Python Environment and Package Exercises"
        />
      </Row>
      <Row>
        <p>
          In this module, students will practice setting up a Python environment
          and installing packages using pip.
        </p>
      </Row>

      <Row>
        <Col md={3}>
          <NavigationMenu links={exerciseLinks} prefix={"/module2/exercise"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={exerciseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default ExercisePython;