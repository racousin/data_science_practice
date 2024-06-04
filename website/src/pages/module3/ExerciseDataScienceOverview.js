import React from "react";
import { Container, Row, Col, Nav } from "react-bootstrap";
import { Routes, Route, Link } from "react-router-dom";
import ExerciseEvaluation from "pages/ExerciseEvaluation";
import Exercise1 from "pages/module3/exercise/Exercise1";
import Exercise2 from "pages/module3/exercise/Exercise2";
// Import other exercises as needed

const ExerciseDataScienceOverview = () => {
  return (
    <Container>
      <h1 className="my-4">Module 3</h1>
      <p>TODO</p>
      <Row>
        <Col md={3}>
          <Nav variant="pills" className="flex-column">
            <Nav.Link as={Link} to="/module3/course">
              Course
            </Nav.Link>
            <Nav.Link as={Link} to="/exercise-evaluation">
              Exercise Evaluation
            </Nav.Link>
            <Nav.Link as={Link} to="/module3/exercise/exercise1">
              Exercise 1
            </Nav.Link>
            <Nav.Link as={Link} to="/module3/exercise/exercise2">
              Exercise 2
            </Nav.Link>
            {/* Add links to other exercises as needed */}
          </Nav>
        </Col>
        <Col md={9}>
          <Routes>
            <Route
              path="/exercise-evaluation"
              element={<ExerciseEvaluation />}
            />
            <Route path="/exercise1" element={<Exercise1 />} />
            <Route path="/exercise2" element={<Exercise2 />} />
            {/* Add routes for other exercises as needed */}
          </Routes>
        </Col>
      </Row>
    </Container>
  );
};

export default ExerciseDataScienceOverview;
