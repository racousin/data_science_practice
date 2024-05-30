import React from "react";
import { BrowserRouter, Routes, Route, Navigate, Link } from "react-router-dom";
import { Navbar, Nav, Container } from "react-bootstrap";
import Home from "./pages/Home";
import StudentsList from "./components/StudentsList";
import Student from "./components/Student";

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
              <Nav.Link as={Link} to="/students">
                Teaching
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
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Container>
    </BrowserRouter>
  );
}

export default App;
