import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const ExerciseCloudIntegration = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1",
      component: lazy(() => import("pages/module8/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module8/exercise/Exercise2")),
    },
    // Add links to other exercises as needed
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={4}
          isCourse={false}
          title="Module 7: ExerciseBuildingEvaluation"
        />
      </Row>
      <Row>
        <p>
          In this module, you will practice retrieving data from different
          sources using Python.
        </p>
      </Row>

      <Row>
        <Col md={3}>
          <NavigationMenu links={exerciseLinks} prefix={"/module8/exercise"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={exerciseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default ExerciseCloudIntegration;
