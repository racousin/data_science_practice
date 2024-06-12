import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { Container, ListGroup, Badge } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faCheckCircle,
  faTimesCircle,
} from "@fortawesome/free-solid-svg-icons";

const Student = () => {
  const { studentId } = useParams();
  const [modulesResults, setmodulesResults] = useState({});
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`/students/${studentId}.json`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load the data");
        }
        return response.json();
      })
      .then((data) => setmodulesResults(data))
      .catch((error) => {
        console.error("Error fetching MODULE results:", error);
        setError("Failed to fetch MODULE results.");
      });
  }, [studentId]);

  const getResultIcon = (result) => {
    if (result.endsWith("success")) {
      return <FontAwesomeIcon icon={faCheckCircle} color="green" />;
    }
    return <FontAwesomeIcon icon={faTimesCircle} color="red" />;
  };

  return (
    <Container>
      <h1>MODULE Results for {studentId}</h1>
      {error && <Badge bg="danger">{error}</Badge>}
      <ListGroup>
        {Object.entries(modulesResults).map(([module, result]) => (
          <ListGroup.Item key={module}>
            {module.toUpperCase()}: {getResultIcon(result)}
          </ListGroup.Item>
        ))}
      </ListGroup>
    </Container>
  );
};

export default Student;
