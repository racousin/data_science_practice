import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseDataCleaningAndPreparation = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Data Cleaning and Preparation",
      component: lazy(() => import("pages/module5/course/Introduction")),
    },
    {
      to: "/handle-missing-values",
      label: "Handle Missing Values",
      component: lazy(() => import("pages/module5/course/HandleMissingValues")),
      subLinks: [
        { id: "what-are-missing-values", label: "What are Missing Values?" },
        { id: "visualize-missing-values", label: "Visualize Missing Values" },
        { id: "imputation", label: "Mean/Median/Mode Imputation" },
        { id: "forward-fill", label: "Forward Fill" },
        { id: "k-nearest-neighbors", label: "K-Nearest Neighbors" },
        { id: "is_missing", label: "Adding a 'is_missing' Column" },
      ],
    },
    {
      to: "/handle-categorical-values",
      label: "Handle Categorical Values",
      component: lazy(() =>
        import("pages/module5/course/HandleCategoricalValues")
      ),
      subLinks: [
        { id: "types-of-categorical-data", label: "Types of Categorical Data" },
        {
          id: "identify-and-visualize-categorical-values",
          label: "Identify and Visualize Categorical Values",
        },
        { id: "one-hot-encoding", label: "One-Hot Encoding" },
        { id: "label-encoding", label: "Label Encoding" },
        {
          id: "handling-unseen-categories",
          label: "Handling Unseen Categories",
        },
        {
          id: "feature-engineering",
          label: "Feature Engineering with Categorical Data",
        },
        {
          id: "use-in-models",
          label: "Using Categorical Data in Models",
        },
      ],
    },
    {
      to: "/handle-duplicates",
      label: "Handle Duplicates",
      component: lazy(() => import("pages/module5/course/HandleDuplicates")),
    },
    {
      to: "/handle-outliers",
      label: "Handle Outliers",
      component: lazy(() => import("pages/module5/course/HandleOutliers")),
    },
    {
      to: "/feature-engineering",
      label: "Feature Engineering",
      component: lazy(() =>
        import("pages/module5/course/FeatureEngineeringTechniques")
      ),
    },
    {
      to: "/correct-inconsistencies",
      label: "Correct Inconsistencies",
      component: lazy(() =>
        import("pages/module5/course/DataQualityAndInconsistencies")
      ),
    },
    {
      to: "/feature-selection",
      label: "Feature Selection",
      component: lazy(() => import("pages/module5/course/FeatureSelection")),
    },
    {
      to: "/scaling-and-normalization",
      label: "Scaling And Normalization",
      component: lazy(() =>
        import("pages/module5/course/ScalingAndNormalization")
      ),
    },
    // {
    //   to: "/filter-outliers",
    //   label: "Filter Outliers",
    //   component: lazy(() => import("pages/module5/course/FilterOutliers")),
    // },
    // {
    //   to: "/best-practices",
    //   label: "Best Practices and Common Pitfalls",
    //   component: lazy(() => import("pages/module5/course/BestPractices")),
    // },
  ];

  const location = useLocation();
  const module = 5;
  return (
    <ModuleFrame
      module={5}
      isCourse={true}
      title="Module 5: Data Cleaning and Preparation"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>In this module, you will learn about #TOOO</p>
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

export default CourseDataCleaningAndPreparation;
