import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseTabularModels = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Tabular Models",
      component: lazy(() => import("pages/module6/course/Introduction")),
      subLinks: [
        {
          id: "definition",
          label: "Definition and characteristics of tabular data",
        },
        {
          id: "supervised-learning",
          label: "Overview of supervised learning for tabular data",
        },
        {
          id: "importance",
          label: "Importance of tabular models in data science",
        },
      ],
    },
    {
      to: "/linear-models",
      label: "Linear Models",
      component: lazy(() => import("pages/module6/course/LinearModels")),
      subLinks: [
        { id: "linear-regression", label: "Linear Regression" },
        { id: "logistic-regression", label: "Logistic Regression" },
        { id: "regularized-models", label: "Regularized Linear Models" },
      ],
    },
    {
      to: "/tree-based-models",
      label: "Tree-based Models",
      component: lazy(() => import("pages/module6/course/TreeBasedModels")),
      subLinks: [
        { id: "decision-trees", label: "Decision Trees" },
        { id: "random-forests", label: "Random Forests" },
        { id: "gradient-boosting", label: "Gradient Boosting Machines" },
      ],
    },
    {
      to: "/svm",
      label: "Support Vector Machines (SVM)",
      component: lazy(() => import("pages/module6/course/SVM")),
      subLinks: [
        { id: "theory", label: "Theory and kernel trick" },
        { id: "types", label: "Linear and non-linear SVMs" },
        {
          id: "implementation",
          label: "Implementation and key hyperparameters",
        },
      ],
    },
    {
      to: "/knn",
      label: "K-Nearest Neighbors (KNN)",
      component: lazy(() => import("pages/module6/course/KNN")),
      subLinks: [
        { id: "theory", label: "Theory and distance metrics" },
        {
          id: "implementation",
          label: "Implementation and key hyperparameters",
        },
        { id: "pros-cons", label: "Pros and cons of lazy learning" },
      ],
    },
    {
      to: "/naive-bayes",
      label: "Naive Bayes",
      component: lazy(() => import("pages/module6/course/NaiveBayes")),
      subLinks: [
        { id: "theory", label: "Theory and types" },
        { id: "implementation", label: "Implementation and use cases" },
        { id: "pros-cons", label: "Pros and cons" },
      ],
    },
    {
      to: "/ensemble-techniques",
      label: "Advanced Ensemble Techniques",
      component: lazy(() => import("pages/module6/course/EnsembleTechniques")),
      subLinks: [
        { id: "bagging", label: "Bagging" },
        { id: "boosting", label: "Boosting" },
        { id: "stacking", label: "Stacking" },
      ],
    },
    {
      to: "/time-series-models",
      label: "Time Series Models",
      component: lazy(() => import("pages/module6/course/TimeSeriesModels")),
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
        { id: "genetic-algorithms", label: "Genetic Algorithms" },
      ],
    },
    {
      to: "/model-selection",
      label: "Model Selection Techniques",
      component: lazy(() => import("pages/module6/course/ModelSelection")),
      subLinks: [
        { id: "cross-validation", label: "Cross-validation strategies" },
        { id: "nested-cv", label: "Nested cross-validation" },
        { id: "comparison-metrics", label: "Model comparison metrics" },
      ],
    },
    {
      to: "/automl",
      label: "AutoML for Tabular Data",
      component: lazy(() => import("pages/module6/course/AutoML")),
      subLinks: [
        { id: "introduction", label: "Introduction to AutoML" },
        { id: "libraries", label: "Popular AutoML libraries" },
        { id: "implementation", label: "Implementing AutoML pipelines" },
        { id: "pros-cons", label: "Pros and cons of AutoML" },
      ],
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
            <p>
              In this module, you will learn about various tabular models used
              in machine learning, from basic linear models to advanced ensemble
              techniques. You'll explore the theory behind these models, their
              implementation using popular libraries like scikit-learn, and
              techniques for model selection and hyperparameter optimization.
              The module concludes with an introduction to AutoML for tabular
              data.
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

export default CourseTabularModels;
