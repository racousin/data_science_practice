import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseGit = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1",
      component: lazy(() => import("pages/module1/exercise/Exercise1")),
    },
  ];

  return (
    <ModuleFrame
      module={1}
      isCourse={false}
      title="Module 1: Git Exercises"
      courseLinks={exerciseLinks}
    >
      <Row>
        <p>
          Practice using Git for version control and GitHub for collaboration.
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

export default ExerciseGit;
