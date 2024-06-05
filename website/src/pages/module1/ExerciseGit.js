import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const ExerciseGit = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1",
      component: lazy(() => import("pages/module1/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module1/exercise/Exercise2")),
    },
    // Add links to other exercises as needed
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={1}
          isCourse={false}
          title="Module 1: Git Exercises"
        />
      </Row>
      <Row>
        <p>
          In this module, students will practice using Git for version control
          and GitHub for collaboration.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={exerciseLinks} prefix={"/module1/exercise"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={exerciseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default ExerciseGit;
