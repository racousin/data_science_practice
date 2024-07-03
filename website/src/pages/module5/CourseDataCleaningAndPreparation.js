import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseDataCleaningAndPreparation = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const DataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handle_inconsistencies.csv";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/handle_inconsistencies.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/handle_inconsistencies.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/handle_inconsistencies.ipynb";
  const metadata = {
    description:
      "This dataset contains demographic information including state, country, age, and date.",
    source: "Demographic Survey Records",
    target: null,
    listData: [
      {
        name: "State",
        description: "The state or province of residence",
      },
      {
        name: "Country",
        description: "The country of residence",
      },
      {
        name: "Age",
        description: "The age of the individual in years",
      },
      {
        name: "Date",
        description: "The date the information was recorded (YYYY-MM-DD)",
      },
    ],
  };
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Data Cleaning and Preparation",
      component: lazy(() => import("pages/module5/course/Introduction")),
    },
    {
      to: "/correct-inconsistencies",
      label: "Handle Inconsistencies",
      component: lazy(() =>
        import("pages/module5/course/HandleInconsistencies")
      ),
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
        { id: "notebook-example", label: "Notebook Example" },
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
        { id: "notebook-example", label: "Notebook Example" },
      ],
    },
    {
      to: "/handle-duplicates",
      label: "Handle Duplicates",
      component: lazy(() => import("pages/module5/course/HandleDuplicates")),
      subLinks: [
        { id: "types-of-duplicates", label: "Types of Duplicates" },
        {
          id: "identifying-duplicates",
          label: "Identifying Duplicates",
        },
        { id: "visualize-duplicates", label: "Visualize Duplicates" },
        { id: "removing-duplicates", label: "Removing Duplicates" },
        {
          id: "advanced-techniques",
          label: "Advanced Techniques",
        },
        {
          id: "considerations",
          label: "Considerations",
        },
        { id: "notebook-example", label: "Notebook Example" },
      ],
    },
    {
      to: "/handle-outliers",
      label: "Handle Outliers",
      component: lazy(() => import("pages/module5/course/HandleOutliers")),
      subLinks: [
        { id: "what-are-outliers", label: "What Are Outliers?" },
        {
          id: "detecting-outliers",
          label: "Detecting Outliers",
        },
        { id: "visualize-outliers", label: "Visualize Outliers" },
        { id: "managing-outliers", label: "Managing Outliers" },
        {
          id: "considerations",
          label: "Considerations",
        },
        { id: "notebook-example", label: "Notebook Example" },
      ],
    },
    {
      to: "/feature-engineering",
      label: "Feature Engineering",
      component: lazy(() =>
        import("pages/module5/course/FeatureEngineeringTechniques")
      ),
    },
    {
      to: "/scaling-and-normalization",
      label: "Scaling And Normalization",
      component: lazy(() =>
        import("pages/module5/course/ScalingAndNormalization")
      ),
    },
    {
      to: "/feature-selection-dimensionality-reduction",
      label: "FeatureSelection And Dimensionality Reduction",
      component: lazy(() =>
        import(
          "pages/module5/course/FeatureSelectionAndDimensionalityReduction"
        )
      ),
    },
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
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
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
    </ModuleFrame>
  );
};

export default CourseDataCleaningAndPreparation;
