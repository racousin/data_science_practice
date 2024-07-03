import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseTextProcessing = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Text Processing",
      component: lazy(() => import("pages/module9/course/Introduction")),
    },
    {
      to: "/text-preprocessing",
      label: "Text Pre-processing",
      component: lazy(() => import("pages/module9/course/TextPreprocessing")),
    },
    {
      to: "/feature-extraction",
      label: "Feature Extraction from Text",
      component: lazy(() => import("pages/module9/course/FeatureExtraction")),
    },
    {
      to: "/text-classification",
      label: "Text Classification",
      component: lazy(() => import("pages/module9/course/TextClassification")),
    },
    {
      to: "/topic-modeling",
      label: "Topic Modeling",
      component: lazy(() => import("pages/module9/course/TopicModeling")),
    },
    {
      to: "/nlp-with-deep-learning",
      label: "Natural Language Processing with Deep Learning",
      component: lazy(() => import("pages/module9/course/NLPWithDeepLearning")),
    },
    {
      to: "/advanced-applications",
      label: "Advanced Applications of Text Processing",
      component: lazy(() =>
        import("pages/module9/course/AdvancedApplications")
      ),
    },
  ];

  const location = useLocation();
  const module = 9;
  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 9: Text Processing"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about text processing techniques
              and applications in data science.
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
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseTextProcessing;
