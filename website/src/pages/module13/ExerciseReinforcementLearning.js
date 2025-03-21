import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const ExerciseReinforcementLearning = () => {
  const exerciseLinks = [
    {
      to: "/exercise0",
      label: "Exercise 0",
      component: lazy(() => import("pages/module13/exercise/Exercise0")),
    },
    {
      to: "/exercise1",
      label: "Exercise 1",
      component: lazy(() => import("pages/module13/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: <>Exercise 2<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module13/exercise/Exercise2")),
    },
    {
      to: "/exercise3",
      label: "Exercise 3",
      component: lazy(() => import("pages/module13/exercise/Exercise3")),
    },
    {
      to: "/exercise4",
      label: <>Exercise 4<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module13/exercise/Exercise4")),
    },
    {
      to: "/exercise5",
      label: "Exercise 5",
      component: lazy(() => import("pages/module13/exercise/Exercise5")),
    },
  ];

  const location = useLocation();
  const module = 13;
  return (
    <ModuleFrame
      module={13}
      isCourse={false}
      title="Module 13: Exercise Reinforcement Learning"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Row>
            <p>
              In this module, you will practice building and applying
              reinforcement learning algorithms.
            </p>
          </Row>
          <Row>
            <Col>
              <p>Last Updated: {"2024-09-20"}</p>
            </Col>
          </Row>
        </>
      )}
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={exerciseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default ExerciseReinforcementLearning;
