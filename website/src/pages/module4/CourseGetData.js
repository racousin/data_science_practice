import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CourseGetData = () => {
  const courseLinks = [
    {
      to: "/files",
      label: "Files",
      component: lazy(() => import("pages/module4/course/Files")),
    },
    {
      to: "/databases",
      label: "Databases",
      component: lazy(() => import("pages/module4/course/Databases")),
    },
    {
      to: "/apis",
      label: "APIs",
      component: lazy(() => import("pages/module4/course/APIs")),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={4}
          isCourse={true}
          title="Module 4: Getting Data"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about the different sources of data and
          how to retrieve data from them using Python.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module4/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseGetData;
