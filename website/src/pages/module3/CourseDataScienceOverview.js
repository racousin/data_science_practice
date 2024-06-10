import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const CourseDataScienceOverview = () => {
  const courseLinks = [
    {
      to: "/jobs-and-evolution",
      label: "Jobs and Evolution",
      component: lazy(() => import("pages/module3/course/JobsAndEvolution")),
    },
    {
      to: "/business-issues",
      label: "Business Issues",
      component: lazy(() => import("pages/module3/course/BusinessIssues")),
    },
    {
      to: "/data-types",
      label: "Data Types",
      component: lazy(() => import("pages/module3/course/DataTypes")),
    },
    {
      to: "/exploratory-data-analysis",
      label: "Exploratory Data Analysis",
      component: lazy(() =>
        import("pages/module3/course/ExploratoryDataAnalysis")
      ),
    },
    {
      to: "/machine-learning-pipeline",
      label: "Machine Learning Pipeline",
      component: lazy(() =>
        import("pages/module3/course/MachineLearningPipeline")
      ),
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
