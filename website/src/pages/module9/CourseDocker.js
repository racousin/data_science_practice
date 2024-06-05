import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CourseDocker = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Containerization",
      component: lazy(() => import("pages/module9/course/Introduction")),
    },
    {
      to: "/docker-fundamentals",
      label: "Docker Fundamentals",
      component: lazy(() => import("pages/module9/course/DockerFundamentals")),
    },
    {
      to: "/working-with-containers",
      label: "Working with Docker Containers",
      component: lazy(() =>
        import("pages/module9/course/WorkingWithContainers")
      ),
    },
    {
      to: "/docker-compose-services",
      label: "Docker Compose and Services",
      component: lazy(() =>
        import("pages/module9/course/DockerComposeServices")
      ),
    },
    {
      to: "/docker-in-development-production",
      label: "Docker in Development and Production",
      component: lazy(() =>
        import("pages/module9/course/DockerInDevelopmentProduction")
      ),
    },
    {
      to: "/advanced-techniques",
      label: "Advanced Docker Techniques",
      component: lazy(() => import("pages/module9/course/AdvancedTechniques")),
    },
    {
      to: "/docker-for-data-science",
      label: "Docker for Data Science and Machine Learning",
      component: lazy(() =>
        import("pages/module9/course/DockerForDataScience")
      ),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation module={9} isCourse={true} title="Module 9: Docker" />
      </Row>
      <Row>
        <p>
          In this module, you will learn about Docker, a platform for building,
          shipping, and running applications in containers.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module9/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseDocker;
