import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

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
    <ModuleFrame
      module={4}
      isCourse={true}
      title="Module 4: Getting Data"
      courseLinks={courseLinks}
    >
      <Row>
        <p>
          In this module, you will learn about the different sources of data and
          how to retrieve data from them using Python.
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

export default CourseGetData;
