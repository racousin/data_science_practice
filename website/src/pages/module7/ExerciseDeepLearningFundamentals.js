import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const ExerciseDeepLearningFundamentals = () => {
  const exerciseLinks = [
    // Add links to other exercises as needed
  ];

  const location = useLocation();
  const module = 7;
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 8: Exercise Deep Learning Fundamentals"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Row>
            <p>
              In this module, you will practice building and training neural
              networks using PyTorch.
            </p>
          </Row>
          <Row>
            <Col>
              <p>Last Updated: {"2024-06-07"}</p>
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

export default ExerciseDeepLearningFundamentals;
