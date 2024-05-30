import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";

const StudentsList = () => {
  const [students, setStudents] = useState([]);

  useEffect(() => {
    fetch("/students/students.json")
      .then((response) => response.json())
      .then((data) => setStudents(data))
      .catch((error) => console.error("Error fetching student list:", error));
  }, []);

  return (
    <div>
      <h1>Student List</h1>
      <ul>
        {students.map((student) => (
          <li key={student.id}>
            <Link to={`/student/${student.id}`}>{student.id}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default StudentsList;
