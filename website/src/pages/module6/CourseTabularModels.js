import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseTabularModels = () => {
  const courseLinks = [
    {
      to: "/model-selection",
      label: "Model Selection Techniques",
      component: lazy(() => import("pages/module6/course/ModelSelection")),
    },
    {
      to: "/hyperparameter-optimization",
      label: "Hyperparameter Optimization",
      component: lazy(() =>
        import("pages/module6/course/HyperparameterOptimization")
      ),
    },
    {
      to: "/models",
      label: "Models",
      component: lazy(() => import("pages/module6/course/Models")),
    },
    {
      to: "/ensemble-models",
      label: "Ensemble Models",
      component: lazy(() => import("pages/module6/course/EnsembleModels")),
    },
    {
      to: "/ensemble-techniques",
      label: "Ensemble Techniques",
      component: lazy(() =>
        import("pages/module6/course/EnsembleTechniques")
      ),
    },
    // {
    //   to: "/time-series-models",
    //   label: "Time Series Models",
    //   component: lazy(() => import("pages/module6/course/TimeSeriesModels")),
    // },
    // {
    //   to: "/automl",
    //   label: "AutoML for Tabular Data",
    //   component: lazy(() => import("pages/module6/course/AutoML")),
    // },
    {
      to: "/CaseStudy6",
      label: "Case Study",
      component: lazy(() => import("pages/module6/course/CaseStudy")),
    },
  ];
  

  const location = useLocation();
  const module = 6;
  return (
    <ModuleFrame
      module={6}
      isCourse={true}
      title="Module 6: Tabular Models"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
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
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseTabularModels;
