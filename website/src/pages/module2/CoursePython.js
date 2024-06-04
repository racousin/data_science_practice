import React from "react";
import { Container, Row, Col, Nav } from "react-bootstrap";
import { Routes, Route, Link } from "react-router-dom";
import Introduction from "pages/module2/course/Introduction";
import SettingUpPythonEnvironment from "pages/module2/course/SettingUpPythonEnvironment";
import InstallingPackages from "pages/module2/course/InstallingPackages";
import BuildingPackages from "pages/module2/course/BuildingPackages";
import InstallPython from "pages/module2/course/InstallPython";
import NotebookAndColab from "./course/NotebookAndColab";
import BestPractices from "./course/BestPractices";

const CoursePython = () => {
  return (
    <Container>
      <h1 className="my-4">Module 2: Python Environment and Package</h1>
      <p>
        In this module, students will learn how to set up a Python environment
        and install packages using pip.
      </p>
      <Row>
        <Col md={3}>
          <Nav variant="pills" className="flex-column">
            <Nav.Link as={Link} to="/module2/course/introduction">
              Introduction
            </Nav.Link>
            <Nav.Link as={Link} to="/module2/course/install-python">
              Install Python
            </Nav.Link>
            <Nav.Link
              as={Link}
              to="/module2/course/setting-up-python-environment"
            >
              Setting Up Python Environment
            </Nav.Link>
            <Nav.Link as={Link} to="/module2/course/installing-packages">
              Installing Packages
            </Nav.Link>
            <Nav.Link as={Link} to="/module2/course/building-packages">
              Building Packages
            </Nav.Link>
            <Nav.Link as={Link} to="/module2/course/notebook-and-colab">
              Notebook And Colab
            </Nav.Link>
            <Nav.Link as={Link} to="/module2/course/best-practices">
              Best Practices
            </Nav.Link>
          </Nav>
        </Col>
        <Col md={9}>
          <Routes>
            <Route path="introduction" element={<Introduction />} />
            <Route path="install-python" element={<InstallPython />} />
            <Route
              path="setting-up-python-environment"
              element={<SettingUpPythonEnvironment />}
            />
            <Route
              path="installing-packages"
              element={<InstallingPackages />}
            />
            <Route path="building-packages" element={<BuildingPackages />} />
            <Route path="notebook-and-colab" element={<NotebookAndColab />} />
            <Route path="best-practices" element={<BestPractices />} />
          </Routes>
        </Col>
      </Row>
    </Container>
  );
};

export default CoursePython;
