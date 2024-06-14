import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseModelBuildingEvaluation = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Model Building",
      component: lazy(() => import("pages/module6/course/Introduction")),
    },
    {
      to: "/regression-models",
      label: "Regression Models",
      component: lazy(() => import("pages/module6/course/RegressionModels")),
    },
    {
      to: "/classification-models",
      label: "Classification Models",
      component: lazy(() =>
        import("pages/module6/course/ClassificationModels")
      ),
    },
    {
      to: "/clustering-dimensionality-reduction",
      label: "Clustering and Dimensionality Reduction",
      component: lazy(() =>
        import("pages/module6/course/ClusteringDimensionalityReduction")
      ),
    },
    {
      to: "/model-evaluation-metrics",
      label: "Model Evaluation Metrics",
      component: lazy(() =>
        import("pages/module6/course/ModelEvaluationMetrics")
      ),
    },
    {
      to: "/model-optimization-tuning",
      label: "Model Optimization and Tuning",
      component: lazy(() =>
        import("pages/module6/course/ModelOptimizationTuning")
      ),
    },
    {
      to: "/deploying-models",
      label: "Deploying Models",
      component: lazy(() => import("pages/module6/course/DeployingModels")),
    },
  ];

  const location = useLocation();
  const module = 6;
  return (
    <ModuleFrame
      module={6}
      isCourse={true}
      title="Module 6: Model Building and Evaluation"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about the process of building and
              evaluating machine learning models.
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

export default CourseModelBuildingEvaluation;
