import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CourseTextProcessing = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Text Processing",
      component: lazy(() => import("pages/module12/course/Introduction")),
    },
    {
      to: "/text-preprocessing",
      label: "Text Pre-processing",
      component: lazy(() => import("pages/module12/course/TextPreprocessing")),
    },
    {
      to: "/feature-extraction",
      label: "Feature Extraction from Text",
      component: lazy(() => import("pages/module12/course/FeatureExtraction")),
    },
    {
      to: "/text-classification",
      label: "Text Classification",
      component: lazy(() => import("pages/module12/course/TextClassification")),
    },
    {
      to: "/topic-modeling",
      label: "Topic Modeling",
      component: lazy(() => import("pages/module12/course/TopicModeling")),
    },
    {
      to: "/nlp-with-deep-learning",
      label: "Natural Language Processing with Deep Learning",
      component: lazy(() =>
        import("pages/module12/course/NLPWithDeepLearning")
      ),
    },
    {
      to: "/advanced-applications",
      label: "Advanced Applications of Text Processing",
      component: lazy(() =>
        import("pages/module12/course/AdvancedApplications")
      ),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={12}
          isCourse={true}
          title="Module 12: Text Processing"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about text processing techniques and
          applications in data science.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module12/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseTextProcessing;
