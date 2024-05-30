import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { ListGroup, Container, Alert } from "react-bootstrap";

const StudentsList = () => {
  const [students, setStudents] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/students/config/students.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => setStudents(data))
      .catch((error) => {
        console.error("Error fetching student list:", error);
        setError("Failed to fetch student data.");
      });
    console.log(students);
  }, []);

  return (
    <Container>
      <h1>Student List</h1>
      {error && <Alert variant="danger">{error}</Alert>}
      <ListGroup>
        {students.map((student) => (
          <ListGroup.Item
            key={student.id}
            action
            as={Link}
            to={`/student/${student.id}`}
          >
            {student.id}
          </ListGroup.Item>
        ))}
      </ListGroup>
    </Container>
  );
};

export default StudentsList;
