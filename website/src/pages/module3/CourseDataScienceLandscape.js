import React, { lazy } from "react";
import { Row, Col } from "react-bootstrap";
import { Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
import { TouchpadOff } from "lucide-react";


// TODO 
// explain hyper clearly the steps of eval and train with a complete example
// 1 train and eval
// 2 train on all and predict

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
        { id: "best-practices", label: "Best Practices: Baseline and Iterate" }
      ],
    },
    {
      to: "/model-training-prediction",
      label: "Model Training and Prediction",
      component: lazy(() => import("pages/module3/course/ModelTrainingPrediction")),
      subLinks: [
        { id: "model-fitting", label: "Training" },
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
      to: "/eda-case-study",
      label: "EDA Case Study",
      component: lazy(() => import("pages/module3/course/CaseStudy")),
    },
    {
      to: "/model-baseline-case-study",
      label: "Model Baseline Case Study",
      component: lazy(() => import("pages/module3/course/CaseStudyML")),
    },
  ];

  const title = `Module 3: Data Science landscape`;
  const location = useLocation();
  const module = 3;

  return (
    <ModuleFrame
      module={3}
      isCourse={true}
      title={title}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Last Updated: {"2024-09-20"}</p>
            </Grid.Col>
          </Grid>
        </>
      )}
      <Grid>
        <Grid.Col span={{ md: 12 }}>
          <DynamicRoutes routes={courseLinks} />
        </Grid.Col>
      </Grid>
    </ModuleFrame>
  );
};

export default CourseDataScienceLandscape;
