import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseFeatureEngineering = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1",
      component: lazy(() => import("pages/module5/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module5/exercise/Exercise2")),
    },
    // Add links to other exercises as needed
  ];

  return (
    <ModuleFrame
      module={5}
      isCourse={false}
      title="Module 5: Exercise Feature Engineering"
      courseLinks={exerciseLinks}
    >
      <Row>
        <p>
          In this module, you will practice feature engineering techniques to
          improve the performance of machine learning models.
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

export default ExerciseFeatureEngineering;
