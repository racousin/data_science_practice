import React from "react";
import { BrowserRouter, Routes, Route, Navigate, Link } from "react-router-dom";
import { Navbar, Nav, Container } from "react-bootstrap";
import Home from "./pages/Home";
import StudentsList from "./pages/StudentsList";
import Student from "./pages/Student";
import Teaching from "./pages/Teaching";
import CourseGit from "pages/module1/CourseGit";
import ExerciseEvaluation from "pages/ExerciseEvaluation";
import ExerciseGit from "pages/module1/ExerciseGit";
import CoursePython from "pages/module2/CoursePython";
import ExercisePython from "pages/module2/ExercisePython";
import CourseDataScienceOverview from "pages/module3/CourseDataScienceOverview";
import ExerciseDataScienceOverview from "pages/module3/ExerciseDataScienceOverview";

import CourseGetData from "pages/module4/CourseGetData";
import ExerciseGetData from "pages/module4/ExerciseGetData";

import "App.css";

function App() {
  return (
    <BrowserRouter>
      <Navbar bg="dark" variant="dark" expand="lg" sticky="top">
        <Container>
          <Navbar.Brand as={Link} to="/">
            Teaching Portal
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link as={Link} to="/">
                Home
              </Nav.Link>
              <Nav.Link as={Link} to="/teaching">
                Teaching
              </Nav.Link>

              <Nav.Link as={Link} to="/students">
                Student
              </Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>
      <Container className="mt-3">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="students" element={<StudentsList />} />
          <Route path="student/:studentId" element={<Student />} />
          <Route path="teaching" element={<Teaching />} />
          <Route path="module1/course/*" element={<CourseGit />} />
          <Route path="module1/exercise/*" element={<ExerciseGit />} />
          <Route path="module2/course/*" element={<CoursePython />} />
          <Route path="module2/exercise/*" element={<ExercisePython />} />
          <Route
            path="module3/course/*"
            element={<CourseDataScienceOverview />}
          />
          <Route
            path="module3/exercise/*"
            element={<ExerciseDataScienceOverview />}
          />
          <Route path="module4/course/*" element={<CourseGetData />} />
          <Route path="module4/exercise/*" element={<ExerciseGetData />} />
          <Route path="exercise-evaluation" element={<ExerciseEvaluation />} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Container>
    </BrowserRouter>
  );
}

export default App;
