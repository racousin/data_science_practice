import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

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
      subLinks: [
        { id: "jupyter-notebooks", label: "Jupyter Notebooks" },
        { id: "google-colab", label: "Google Colab" },
      ],
    },
    {
      to: "/best-practices",
      label: "Best Practices",
      component: lazy(() => import("pages/module2/course/BestPractices")),
    },
  ];
  return (
    <ModuleFrame
      module={2}
      isCourse={true}
      title="Module 2: Python Environment and Package"
      courseLinks={courseLinks}
    >
      <Row>
        <p>
          In this module, students will learn how to set up a Python environment
          and install packages using pip.
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

export default CoursePython;
