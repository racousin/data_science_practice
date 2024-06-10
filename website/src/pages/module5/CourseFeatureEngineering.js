import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const CourseFeatureEngineering = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Feature Engineering",
      component: lazy(() => import("pages/module5/course/Introduction")),
    },
    {
      to: "/data-preprocessing",
      label: "Data Preprocessing Techniques",
      component: lazy(() => import("pages/module5/course/DataPreprocessing")),
    },
    {
      to: "/feature-extraction",
      label: "Feature Extraction and Transformation",
      component: lazy(() => import("pages/module5/course/FeatureExtraction")),
    },
    {
      to: "/feature-selection",
      label: "Feature Selection Techniques",
      component: lazy(() => import("pages/module5/course/FeatureSelection")),
    },
    {
      to: "/advanced-techniques",
      label: "Advanced Feature Engineering Techniques",
      component: lazy(() => import("pages/module5/course/AdvancedTechniques")),
    },
    {
      to: "/best-practices",
      label: "Best Practices and Common Pitfalls",
      component: lazy(() => import("pages/module5/course/BestPractices")),
    },
  ];

  return (
    <ModuleFrame
      module={5}
      isCourse={true}
      title="Module 5: Feature Engineering"
      courseLinks={courseLinks}
    >
      <Row>
        <p>
          In this module, you will learn about the process of feature
          engineering, which is crucial for improving the performance of machine
          learning models.
        </p>
      </Row>
      <Row>
        <Col>
          <p>Last Updated: {"2024-06-07"}</p>
        </Col>
      </Row>
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseFeatureEngineering;
