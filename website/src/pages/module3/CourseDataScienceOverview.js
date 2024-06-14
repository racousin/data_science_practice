import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const CourseDataScienceOverview = () => {
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
      to: "/note-bookand-colab",
      label: "Notebook And Colab",
      component: lazy(() => import("pages/module3/course/NotebookAndColab")),
      subLinks: [
        { id: "jupyter-notebooks", label: "Jupyter Notebooks" },
        { id: "google-colab", label: "Google Colab" },
      ],
    },
    {
      to: "/exploratory-data-analysis",
      label: "Exploratory Data Analysis",
      component: lazy(() =>
        import("pages/module3/course/ExploratoryDataAnalysis")
      ),
      subLinks: [
        { id: "jupyter-notebooks", label: "Jupyter Notebooks" },
        { id: "google-colab", label: "Google Colab" },
      ],
    },
    {
      to: "/best-practices-and-resources",
      label: "Best Practices And Resources",
      component: lazy(() =>
        import("pages/module3/course/BestPracticesAndRessources.js")
      ),
      subLinks: [
        {
          id: "installing-packages-in-colab",
          label: "Installing Packages in Colab",
        },
        {
          id: "using-gpu-in-colab",
          label: "Using GPU and Other Resources in Colab",
        },
        {
          id: "bi-tools",
          label: "BI Tools",
        },
        {
          id: "resources",
          label: "Useful Links and Resources",
        },
      ],
    },
  ];
  const title = `Module 3: Data Science Overview`;

  return (
    <ModuleFrame
      module={3}
      isCourse={true}
      title={title}
      courseLinks={courseLinks}
    >
      <Row>
        <p>
          In this module, you will learn about the jobs and evolution of data
          science, the business issues it can answer, the types of data used in
          data science, exploratory data analysis, and machine learning
          pipelines.
        </p>
      </Row>
      <Row>
        <Col>
          <p>Last Updated: {"2024-06-07"}</p>
        </Col>
      </Row>
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseDataScienceOverview;
