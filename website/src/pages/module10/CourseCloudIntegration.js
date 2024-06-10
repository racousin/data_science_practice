import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const CourseCloudIntegration = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Cloud Computing and GCP",
      component: lazy(() => import("pages/module10/course/Introduction")),
    },
    {
      to: "/managing-compute-resources",
      label: "Managing Compute Resources",
      component: lazy(() =>
        import("pages/module10/course/ManagingComputeResources")
      ),
    },
    {
      to: "/storing-managing-data",
      label: "Storing and Managing Data",
      component: lazy(() =>
        import("pages/module10/course/StoringManagingData")
      ),
    },
    {
      to: "/data-analysis-machine-learning",
      label: "Data Analysis and Machine Learning on GCP",
      component: lazy(() =>
        import("pages/module10/course/DataAnalysisMachineLearning")
      ),
    },
    {
      to: "/networking-security",
      label: "Networking and Security",
      component: lazy(() => import("pages/module10/course/NetworkingSecurity")),
    },
    {
      to: "/devops-in-the-cloud",
      label: "DevOps in the Cloud",
      component: lazy(() => import("pages/module10/course/DevOpsInTheCloud")),
    },
    {
      to: "/architecting-scalable-applications",
      label: "Architecting Scalable Applications",
      component: lazy(() =>
        import("pages/module10/course/ArchitectingScalableApplications")
      ),
    },
  ];

  return (
    <ModuleFrame
      module={10}
      isCourse={true}
      title="Module 10: Cloud Integration (with GCP)"
      courseLinks={courseLinks}
    >
      <Row>
        <p>
          In this module, you will learn about integrating your applications
          with Google Cloud Platform (GCP).
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

export default CourseCloudIntegration;
