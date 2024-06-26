import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseDataCollection = () => {
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

  const location = useLocation();
  const module = 4;
  return (
    <ModuleFrame
      module={4}
      isCourse={true}
      title="Module 4: Data Collection"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about the different sources of data
              and how to retrieve data from them using Python.
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

export default CourseDataCollection;
