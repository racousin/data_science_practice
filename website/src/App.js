import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import StudentsList from "./components/StudentsList"; // Adjust the import paths as needed
import Student from "./components/Student";

function App() {
  return (
    <div>
      <Routes>
        <Route path="/" element={<Navigate to="/students" replace />} />
        <Route path="students" element={<StudentsList />} />
        <Route path="student/:studentId" element={<Student />} />
      </Routes>
    </div>
  );
}

export default App;
