import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseDataScienceOverview = () => {
  const exerciseLinks = [
    {
      to: "/course",
      label: "Course",
      component: lazy(() => import("pages/module3/CourseDataScienceOverview")),
    },
    {
      to: "/exercise-evaluation",
      label: "Exercise Evaluation",
      component: lazy(() => import("pages/ExerciseEvaluation")),
    },
    {
      to: "/exercise1",
      label: "Exercise 1",
      component: lazy(() => import("pages/module3/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module3/exercise/Exercise2")),
    },
    // Add links to other exercises as needed
  ];

  return (
    <ModuleFrame
      module={3}
      isCourse={false}
      title="Module 3"
      courseLinks={exerciseLinks}
    >
      <Row>
        <p>TODO</p>
      </Row>
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={exerciseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default ExerciseDataScienceOverview;
