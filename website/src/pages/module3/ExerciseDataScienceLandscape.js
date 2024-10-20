import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const ExerciseMLPipelineAndExploratoryDataAnalysis = () => {
  const exerciseLinks = [
    {
      to: "/exercise0",
      label: <>Exercise 0<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module3/exercise/Exercise0")),
    },
    {
      to: "/exercise1",
      label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module3/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module3/exercise/Exercise2")),
    },
  ];

  const location = useLocation();
  const module = 3;
  return (
    <ModuleFrame
      module={3}
      isCourse={false}
      title="Module 3"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Row>
            Exercises to perform exploratory data analysis and model
            baseline,using Python and Jupyter Notebooks.
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

export default ExerciseMLPipelineAndExploratoryDataAnalysis;
