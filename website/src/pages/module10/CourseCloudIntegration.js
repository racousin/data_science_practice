import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

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
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={10}
          isCourse={true}
          title="Module 10: Cloud Integration (with GCP)"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about integrating your applications
          with Google Cloud Platform (GCP).
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module10/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseCloudIntegration;
