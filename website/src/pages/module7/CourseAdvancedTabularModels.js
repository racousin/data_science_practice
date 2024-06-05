import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CourseAdvancedTabularModels = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Advanced Modeling Techniques",
      component: lazy(() => import("pages/module7/course/Introduction")),
    },
    {
      to: "/ensemble-methods",
      label: "Ensemble Methods",
      component: lazy(() => import("pages/module7/course/EnsembleMethods")),
    },
    {
      to: "/deep-learning-tabular-data",
      label: "Deep Learning for Tabular Data",
      component: lazy(() =>
        import("pages/module7/course/DeepLearningTabularData")
      ),
    },
    {
      to: "/bayesian-optimization",
      label: "Bayesian Optimization",
      component: lazy(() =>
        import("pages/module7/course/BayesianOptimization")
      ),
    },
    {
      to: "/model-interpretability",
      label: "Model Interpretability",
      component: lazy(() =>
        import("pages/module7/course/ModelInterpretability")
      ),
    },
    {
      to: "/automl-tabular-data",
      label: "AutoML for Tabular Data",
      component: lazy(() => import("pages/module7/course/AutoMLTabularData")),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={7}
          isCourse={true}
          title="Module 7: Advanced Tabular Models"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about advanced modeling techniques that
          are particularly effective for tabular data.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module7/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseAdvancedTabularModels;
