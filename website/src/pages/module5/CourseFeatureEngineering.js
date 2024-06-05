import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

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
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={5}
          isCourse={true}
          title="Module 5: Feature Engineering"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about the process of feature
          engineering, which is crucial for improving the performance of machine
          learning models.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module5/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseFeatureEngineering;
