import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";

const Student = () => {
  const { studentId } = useParams();
  const [studentDetails, setStudentDetails] = useState({ tp: [] });

  useEffect(() => {
    fetch(`/students/${studentId}.json`)
      .then((response) => response.json())
      .then((data) => setStudentDetails(data))
      .catch((error) =>
        console.error("Error fetching student details:", error)
      );
  }, [studentId]);

  return (
    <div>
      <h1>Student Details: {studentDetails.name}</h1>
      <ul>
        {studentDetails.tp.map((item, index) => (
          <li key={index}>
            {item.name}: {item.mark}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Student;
