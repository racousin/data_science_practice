import React, { lazy } from "react";
import { Row, Col } from 'react-bootstrap';
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const CourseTabularModels = () => {
  const courseLinks = [
    {
      to: "/model-selection",
      label: "Model Selection",
      component: lazy(() => import("pages/module6/course/ModelSelection")),
      subLinks: [
        { id: "train-test-split", label: "Train-Test Split" },
        { id: "cross-validation", label: "Cross-Validation Methods" },
        { id: "model-comparison", label: "Model Comparison Example" },
        { id: "best-practices", label: "Best Practices" }
      ]
    },
    {
      to: "/hyperparameter-optimization",
      label: "Hyperparameter Optimization",
      component: lazy(() =>
        import("pages/module6/course/HyperparameterOptimization")
      ),
      subLinks: [
        { id: "grid-search", label: "Grid Search" },
        { id: "random-search", label: "Random Search" },
        { id: "bayesian-optimization", label: "Bayesian Optimization" },
        { id: "scoring-methods", label: "Scoring Methods" },
        { id: "cv-strategies", label: "Cross-validation Strategies" },
        { id: "best-practices", label: "Best Practices" }
      ]
    },
    {
      to: "/models",
      label: "Models",
      component: lazy(() => import("pages/module6/course/Models")),
      subLinks: [
        { id: "linear-models", label: "Linear Models" },
        { id: "knn", label: "K-Nearest Neighbors (KNN)" },
        { id: "svm", label: "Support Vector Machines (SVM)" },
        { id: "decision-trees", label: "Decision Trees" }
      ]
    },
    {
      to: "/ensemble-models",
      label: "Ensemble Models",
      component: lazy(() => import("pages/module6/course/EnsembleModels")),
      subLinks: [
        { id: "boosting", label: "Boosting" },
        { id: "random-forests", label: "Random Forests" },
        { id: "advanced-gradient-boosting", label: "Advanced Gradient Boosting" }
      ]
    },
    {
      to: "/ensemble-techniques",
      label: "Ensemble Techniques",
      component: lazy(() =>
        import("pages/module6/course/EnsembleTechniques")
      ),
      subLinks: [
        { id: "bagging", label: "Bagging (Bootstrap Aggregating)" },
        { id: "stacking", label: "Stacking" }
      ]
    },
    {
      to: "/time-series-models",
      label: "Time Series Models",
      component: lazy(() => import("pages/module6/course/TimeSeriesModels")),
      subLinks: [
        { id: "arima", label: "ARIMA Models" },
        { id: "prophet", label: "Prophet" },
        { id: "lstm", label: "Long Short-Term Memory (LSTM) Networks" }
      ]
    },
    {
      to: "/automl",
      label: "AutoML for Tabular Data",
      component: lazy(() => import("pages/module6/course/AutoML")),
      subLinks: []
    },
    {
      to: "/custom-objectives",
      label: "Custom Objectives Guide",
      component: lazy(() => import("pages/module6/course/CustomObjectivesGuide")),
      subLinks: [
        { id: "understanding-objective", label: "Understanding the Objective" },
        { id: "customization-techniques", label: "Customization Techniques" },
        { id: "model-combinations", label: "Model Combinations" }
      ]
    },
    {
      to: "/CaseStudy6",
      label: "Case Study",
      component: lazy(() => import("pages/module6/course/CaseStudy")),
      subLinks: []
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