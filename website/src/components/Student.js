import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { Container, ListGroup, Badge } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faCheckCircle,
  faTimesCircle,
} from "@fortawesome/free-solid-svg-icons";

const StudentPage = () => {
  const { studentId } = useParams();
  const [tpsResults, setTpsResults] = useState({});
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`/students/${studentId}.json`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load the data");
        }
        return response.json();
      })
      .then((data) => setTpsResults(data))
      .catch((error) => {
        console.error("Error fetching TP results:", error);
        setError("Failed to fetch TP results.");
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
      <h1>TP Results for {studentId}</h1>
      {error && <Badge bg="danger">{error}</Badge>}
      <ListGroup>
        {Object.entries(tpsResults).map(([tp, result]) => (
          <ListGroup.Item key={tp}>
            {tp.toUpperCase()}: {getResultIcon(result)}
          </ListGroup.Item>
        ))}
      </ListGroup>
    </Container>
  );
};

export default StudentPage;
