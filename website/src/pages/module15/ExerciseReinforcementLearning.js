import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseReinforcementLearning = () => {
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
    <ModuleFrame
      module={15}
      isCourse={false}
      title="Module 15: Exercise Reinforcement Learning"
      courseLinks={exerciseLinks}
    >
      <Row>
        <p>
          In this module, you will practice building and applying reinforcement
          learning algorithms in AI.
        </p>
      </Row>
      <Row>
        <Col>
          <p>Last Updated: {"2024-06-07"}</p>
        </Col>
      </Row>
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={exerciseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default ExerciseReinforcementLearning;
