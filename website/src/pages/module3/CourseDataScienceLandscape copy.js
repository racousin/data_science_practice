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
        { id: "data-collection", label: "Data Collection" },
        { id: "data-cleaning", label: "Data Cleaning and Preparation" },
        { id: "feature-engineering", label: "Feature Engineering" },
        { id: "model-building", label: "Model Building" },
        { id: "evaluation", label: "Evaluation" },
        { id: "deployment", label: "Deployment" },
        { id: "monitoring", label: "Monitoring and Maintenance" },
      ],
    },
    {
      to: "/exploratory-data-analysis",
      label: "Exploratory Data Analysis And Model Baseline",
      component: lazy(() =>
        import("pages/module3/course/ExploratoryDataAnalysisAndModelBaseline")
      ),
      subLinks: [
        {
          id: "importance-objectives",
          label: "Importance and Objectives of EDA",
        },
        { id: "eda-techniques", label: "EDA Techniques" },
        { id: "visualization-tools", label: "Visualization Tools for EDA" },
        { id: "statistical-measures", label: "Statistical Measures in EDA" },
        {
          id: "handling-data-issues",
          label: "Handling Missing Data and Outliers",
        },
        { id: "eda-data-types", label: "EDA for Different Data Types" },
        { id: "jupyter-notebooks", label: "Jupyter Notebooks" },
        { id: "google-colab", label: "Google Colab" },
        { id: "case-study", label: "EDA and Model Baseline Case Study" },
      ],
    },
    {
      to: "/model-evaluation-validation",
      label: "Model Evaluation and Validation",
      component: lazy(() =>
        import("pages/module3/course/ModelEvaluationValidation")
      ),
      subLinks: [
        { id: "importance", label: "Importance of Model Evaluation" },
        {
          id: "overfitting-underfitting",
          label: "Overfitting and Underfitting",
        },
        { id: "train-test-splits", label: "Train-Test-Validation Splits" },
        { id: "evaluation-metrics", label: "Evaluation Metrics" },
        { id: "model-comparison", label: "Model Comparison and Selection" },
        {
          id: "imbalanced-datasets",
          label: "Dealing with Imbalanced Datasets",
        },
        {
          id: "model-interpretability",
          label: "Model Interpretability and Explainability",
        },
      ],
    },
    {
      to: "/model-deployment",
      label: "From Model Evaluation to Deployment",
      component: lazy(() => import("pages/module3/course/ModelDeployment")),
      subLinks: [
        {
          id: "versioning-tracking",
          label: "Model Versioning and Experiment Tracking",
        },
        { id: "ab-testing", label: "A/B Testing in Production" },
        { id: "performance-monitoring", label: "Monitoring Model Performance" },
        {
          id: "update-strategies",
          label: "Strategies for Model Updates and Retraining",
        },
      ],
    },
    {
      to: "/best-practices-and-resources",
      label: "Best Practices And Resources",
      component: lazy(() =>
        import("pages/module3/course/BestPracticesAndRessources.js")
      ),
      subLinks: [
        { id: "bi-tools", label: "BI Tools" },
        { id: "resources", label: "Useful Links and Resources" },
      ],
    },
  ];

  const title = `Module 3: ML landscape`;
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
            <p>
              In this module, you will learn about the jobs and evolution of
              data science, the business issues it can answer, the types of data
              used in data science, exploratory data analysis, machine learning
              pipelines, model evaluation, and deployment strategies.
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

export default CourseDataScienceLandscape;
