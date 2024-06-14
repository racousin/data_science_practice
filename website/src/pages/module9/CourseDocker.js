import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

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

  const location = useLocation();
  const module = 9;
  return (
    <ModuleFrame
      module={9}
      isCourse={true}
      title="Module 9: Docker"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about Docker, a platform for
              building, shipping, and running applications in containers.
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

export default CourseDocker;
