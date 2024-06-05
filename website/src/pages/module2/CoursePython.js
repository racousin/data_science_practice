import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CoursePython = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction",
      component: lazy(() => import("pages/module2/course/Introduction")),
    },
    {
      to: "/install-python",
      label: "Install Python",
      component: lazy(() => import("pages/module2/course/InstallPython")),
    },
    {
      to: "/setting-up-python-environment",
      label: "Setting Up Python Environment",
      component: lazy(() =>
        import("pages/module2/course/SettingUpPythonEnvironment")
      ),
    },
    {
      to: "/installing-packages",
      label: "Installing Packages",
      component: lazy(() => import("pages/module2/course/InstallingPackages")),
    },
    {
      to: "/building-packages",
      label: "Building Packages",
      component: lazy(() => import("pages/module2/course/BuildingPackages")),
    },
    {
      to: "/notebook-and-colab",
      label: "Notebook And Colab",
      component: lazy(() => import("pages/module2/course/NotebookAndColab")),
    },
    {
      to: "/best-practices",
      label: "Best Practices",
      component: lazy(() => import("pages/module2/course/BestPractices")),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={2}
          isCourse={true}
          title="Module 2: Python Environment and Package"
        />
      </Row>
      <Row>
        <p>
          In this module, students will learn how to set up a Python environment
          and install packages using pip.
        </p>
      </Row>

      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module2/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CoursePython;
