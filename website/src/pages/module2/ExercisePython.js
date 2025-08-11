import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Container, Grid, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const ExercisePython = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module2/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module2/exercise/Exercise2")),
    },
  ];

  const location = useLocation();
  const module = 2;
  return (
    <ModuleFrame
      module={2}
      isCourse={false}
      title="Module 2: Python Environment and Package Exercises"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
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

export default ExercisePython;
