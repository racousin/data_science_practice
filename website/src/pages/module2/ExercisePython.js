import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExercisePython = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1",
      component: lazy(() => import("pages/module2/exercise/Exercise1")),
    },
  ];

  return (
    <ModuleFrame
      module={2}
      isCourse={false}
      title="Module 2: Python Environment and Package Exercises"
      courseLinks={exerciseLinks}
    >
      <Row>
        <p>
          In this module, students will practice setting up a Python environment
          and installing packages using pip.
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

export default ExercisePython;
