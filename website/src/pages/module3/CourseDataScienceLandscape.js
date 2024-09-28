import React, { lazy } from "react";
import { Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseDataScienceLandscape = () => {
  const courseLinks = [
    {
      to: "/Introduction",
      label: "Introduction",
      component: lazy(() => import("pages/module3/course/Introduction")),
      subLinks: [
        { id: "data", label: "The Data" },
        { id: "applications", label: "The Applications" },
        { id: "roles", label: "Roles in Data Science" },
        { id: "tools", label: "The Data Science Tools" },
      ],
    },
    {
      to: "/machine-learning-pipeline",
      label: "Machine Learning Pipeline",
      component: lazy(() =>
        import("pages/module3/course/MachineLearningPipeline")
      ),
      subLinks: [
        {id: "problem-definition", label: "Problem Definition"},
        { id: "data-collection", label: "Data Collection" },
        { id: "data-cleaning", label: "Data Preprocessing and Feature Engineering" },
        { id: "model-building", label: "Model Selection, Training, and Evaluation" },
        { id: "deployment", label: "Deployment, Monitoring, and Maintenance" },
        { id: "monitoring", label: "Model Interpretability and Explainability" },
      ],
    },
    {
      to: "/model-training-prediction",
      label: "Model Training and Prediction",
      component: lazy(() => import("pages/module3/course/ModelTrainingPrediction")),
      subLinks: [
        { id: "model-fitting", label: "Model Fitting" },
        { id: "prediction", label: "Prediction" },
      ],
    },
    {
      to: "/model-evaluation-validation",
      label: "Model Evaluation",
      component: lazy(() =>
        import("pages/module3/course/ModelEvaluationValidation")
      ),
      subLinks: [
        { id: "performance-metrics", label: "Performance Metrics" },
        { id: "overfitting-underfitting", label: "Overfitting and Underfitting" },
        { id: "bias-variance", label: "Bias-Variance Tradeoff" },
        { id: "cross-validation", label: "Cross-Validation" },
        { id: "time-series-cv", label: "Time Series Cross-Validation" },
      ],
    },
    {
      to: "/evaluation-metrics",
      label: "Evaluation Metrics",
      component: lazy(() => import("pages/module3/course/EvaluationMetrics")),
      subLinks: [
        { id: "regression-metrics", label: "Regression Metrics" },
        { id: "binary-classification-metrics", label: "Binary Classification Metrics" },
        { id: "multi-class-classification-metrics", label: "Multi-class Classification Metrics" },
        { id: "ranking-metrics", label: "Ranking Metrics" },
        { id: "time-series-metrics", label: "Time Series Metrics" },
        { id: "choosing-metrics", label: "Choosing the Right Metric" },
      ],
    },

    {
      to: "/exploratory-data-analysis",
      label: "Exploratory Data Analysis",
      component: lazy(() => import("pages/module3/course/ExploratoryDataAnalysis")),
      subLinks: [
        { id: "main-components", label: "Main Components of EDA" },
        { id: "jupyter-notebooks", label: "Jupyter Notebooks" },
        { id: "google-colab", label: "Google Colab" },
      ],
    },
    {
      to: "/eda-and-model-baseline-case-study",
      label: "EDA and Model Baseline Case Study",
      component: lazy(() => import("pages/module3/course/CaseStudy")),
    },
    // {
    //   to: "/model-deployment",
    //   label: "From Model Evaluation to Deployment",
    //   component: lazy(() => import("pages/module3/course/ModelDeployment")),
    //   subLinks: [
    //     {
    //       id: "versioning-tracking",
    //       label: "Model Versioning and Experiment Tracking",
    //     },
    //     { id: "ab-testing", label: "A/B Testing in Production" },
    //     { id: "performance-monitoring", label: "Monitoring Model Performance" },
    //     {
    //       id: "update-strategies",
    //       label: "Strategies for Model Updates and Retraining",
    //     },
    //   ],
    // },
    // {
    //   to: "/best-practices-and-resources",
    //   label: "Best Practices And Resources",
    //   component: lazy(() =>
    //     import("pages/module3/course/BestPracticesAndRessources.js")
    //   ),
    //   subLinks: [
    //     { id: "bi-tools", label: "BI Tools" },
    //     { id: "resources", label: "Useful Links and Resources" },
    //   ],
    // },
  ];

  const title = `Module 3: Data Science landscape`;
  const location = useLocation();
  const module = 3;

  return (
    <ModuleFrame
      module={3}
      isCourse={true}
      title={title}
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

export default CourseDataScienceLandscape;
