import React from "react";
import { Container, Row, Col, Nav } from "react-bootstrap";
import { Routes, Route, Link } from "react-router-dom";
import JobsAndEvolution from "./course/JobsAndEvolution";
import BusinessIssues from "./course/BusinessIssues";
import DataTypes from "./course/DataTypes";
import ExploratoryDataAnalysis from "./course/ExploratoryDataAnalysis";
import MachineLearningPipeline from "./course/MachineLearningPipeline";

const CourseDataScienceOverview = () => {
  return (
    <Container>
      <h1 className="my-4">Module 3: Data Science Overview</h1>
      <p>
        In this module, you will learn about the jobs and evolution of data
        science, the business issues it can answer, the types of data used in
        data science, exploratory data analysis, and machine learning pipelines.
      </p>
      <Row>
        <Col md={3}>
          <Nav variant="pills" className="flex-column">
            <Nav.Link as={Link} to="/module3/course/jobs-and-evolution">
              Jobs and Evolution
            </Nav.Link>
            <Nav.Link as={Link} to="/module3/course/business-issues">
              Business Issues
            </Nav.Link>
            <Nav.Link as={Link} to="/module3/course/data-types">
              Data Types
            </Nav.Link>
            <Nav.Link as={Link} to="/module3/course/exploratory-data-analysis">
              Exploratory Data Analysis
            </Nav.Link>
            <Nav.Link as={Link} to="/module3/course/machine-learning-pipeline">
              Machine Learning Pipeline
            </Nav.Link>
          </Nav>
        </Col>
        <Col md={9}>
          <Routes>
            <Route path="jobs-and-evolution" element={<JobsAndEvolution />} />
            <Route path="business-issues" element={<BusinessIssues />} />
            <Route path="data-types" element={<DataTypes />} />
            <Route
              path="exploratory-data-analysis"
              element={<ExploratoryDataAnalysis />}
            />
            <Route
              path="machine-learning-pipeline"
              element={<MachineLearningPipeline />}
            />
          </Routes>
        </Col>
      </Row>
    </Container>
  );
};

export default CourseDataScienceOverview;
