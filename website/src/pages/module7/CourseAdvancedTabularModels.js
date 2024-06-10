import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

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
    <ModuleFrame
      module={7}
      isCourse={true}
      title="Module 7: Advanced Tabular Models"
      courseLinks={courseLinks}
    >
      <Row>
        <p>
          In this module, you will learn about advanced modeling techniques that
          are particularly effective for tabular data.
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

export default CourseAdvancedTabularModels;
