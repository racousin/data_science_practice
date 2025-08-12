import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const CourseDataPreprocessing = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction",
      component: lazy(() => import("pages/data-science-practice/module5/course/Introduction")
      ),
      subLinks: [
      ],
    },
    {
      to: "/correct-inconsistencies",
      label: "Handle Inconsistencies",
      component: lazy(() => import("pages/data-science-practice/module5/course/HandleInconsistencies")
      ),
      subLinks: [
        { id: "types-of-inconsistencies", label: "Types of Inconsistencies" },
        { id: "detecting-inconsistencies", label: "Detecting Inconsistencies" },
        {
          id: "solutions-to-inconsistencies",
          label: "Solutions to Inconsistencies",
        },
        { id: "notebook-example", label: "Notebook Example" },
      ],
    },
    {
      to: "/handle-duplicates",
      label: "Handle Duplicates",
      component: lazy(() => import("pages/data-science-practice/module5/course/HandleDuplicates")),
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
      to: "/handle-missing-values",
      label: "Handle Missing Values",
      component: lazy(() => import("pages/data-science-practice/module5/course/HandleMissingValues")),
      subLinks: [
        { id: "types-of-missing-values", label: "Types of Missing Values" },
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
      component: lazy(() => import("pages/data-science-practice/module5/course/HandleCategoricalValues")
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
          id: "advanced-techniques",
          label: "Advanced Techniques",
        },
        {
          id: "considerations-and-best-practices",
          label: "Considerations and Best Practices",
        },
        { id: "notebook-example", label: "Notebook Example" },
      ],
    },
    {
      to: "/handle-outliers",
      label: "Handle Outliers",
      component: lazy(() => import("pages/data-science-practice/module5/course/HandleOutliers")),
      subLinks: [
        { id: "types-of-outliers", label: "Types of Outliers" },
        {
          id: "detecting-visualizing-outliers",
          label: "Detecting and Visualizing Outliers",
        },
        { id: "managing-outliers", label: "Managing Outliers" },
        { id: "considerations-best-practices", label: "Considerations and Best Practices" },
        { id: "notebook-example", label: "Notebook Example" },
      ],
    },
    {
      to: "/feature-engineering",
      label: "Feature Engineering",
      component: lazy(() => import("pages/data-science-practice/module5/course/FeatureEngineering")
      ),
      subLinks: [
        { id: "Decomposition and Feature Extraction", label: "decomposition-extraction" },
        {
          id: "mathematical-transformations",
          label: "Mathematical Transformations",
        },
        {
          id: "binning-aggregation",
          label: "Binning and Aggregation",
        },
        { id: "time-series-features", label: "Time Series Features" },
        {
          id: "considerations-best-practices",
          label: "Considerations and Best Practices",
        },
        { id: "notebook-example", label: "Notebook Example" },
      ],
    },
    {
      to: "/scaling-and-normalization",
      label: "Scaling And Normalization",
      component: lazy(() => import("pages/data-science-practice/module5/course/ScalingAndNormalization")
      ),
      subLinks: [
        { id: "why-scale-normalize", label: "Why Scale and Normalize?" },
        {
          id: "scaling-methods",
          label: "Scaling Methods",
        },
        {
          id: "normalization-methods",
          label: "Normalization Methods",
        },
        { id: "choosing-method", label: "Choosing the Right Method" },
        { id: "practical-considerations", label: "Practical Considerations" },
        { id: "notebook-example", label: "Notebook Example" },
      ],
    },
    {
      to: "/feature-selection-dimensionality-reduction",
      label: "Feature Selection And Dimensionality Reduction",
      component: lazy(() =>
        import(
          "pages/data-science-practice/module5/course/FeatureSelectionAndDimensionalityReduction"
        )
      ),
      subLinks: [
        { id: "feature-selection", label: "Feature Selection Techniques" },
        {
          id: "dimensionality-reduction",
          label: "Dimensionality Reduction Techniques",
        },
        { id: "comparison", label: "Comparison of Techniques" },
        { id: "best-practices", label: "Best Practices" },
        { id: "notebook-example", label: "Notebook Example" },
      ],
    }
  ];
  const location = useLocation();
  const module = 5;
  return (
    <ModuleFrame
      module={5}
      isCourse={true}
      title="Module 5: Data Preprocessing"
      courseLinks={courseLinks}
    >
      <Grid>
        <Grid.Col span={{ md: 11 }}>
          <DynamicRoutes routes={courseLinks} type="course" />
        </Grid.Col>
      </Grid>
      {location.pathname === `/module${module}/course` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Last Updated: {"2024-09-20"}</p>
            </Grid.Col>
          </Grid>
        </>
      )}
    </ModuleFrame>
  );
};
export default CourseDataPreprocessing;
