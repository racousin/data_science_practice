import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Generic module components
import GenericModuleCourse from '../components/GenericModuleCourse';
import GenericModuleExercise from '../components/GenericModuleExercise';

// Special pages
import PrerequistAndMethodologie from '../pages/data-science-practice/module0/PrerequistAandMethodologie';
import ProjectPages from '../pages/data-science-practice/ProjectPages';
import ProjectPage2024 from '../pages/data-science-practice/project-pages/ProjectPage2024';
import ProjectPage2025 from '../pages/data-science-practice/project-pages/ProjectPage2025';
import PermutedMNIST from '../pages/data-science-practice/project-pages/PermutedMNIST';
import BipedalWalker from '../pages/data-science-practice/project-pages/BipedalWalker';
import RepositoriesList from '../pages/data-science-practice/RepositoriesList';
import StudentsList from '../pages/data-science-practice/StudentsList';
import Student from '../pages/data-science-practice/Student';
import StudentProject from '../pages/data-science-practice/StudentProject';

// Course overview page
import CourseOverview from './DataSciencePracticeOverview';

const DataSciencePractice = () => {
  const courseId = 'data-science-practice';
  
  return (
    <Routes>
      <Route path="/" element={<CourseOverview />} />
      
      {/* Module 0 - Prerequisites */}
      <Route path="module0" element={<PrerequistAndMethodologie />} />
      
      {/* Modules 1-15 using generic components */}
      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].map(num => (
        <React.Fragment key={num}>
          <Route path={`module${num}`} element={<GenericModuleCourse courseId={courseId} />} />
          <Route path={`module${num}/course/*`} element={<GenericModuleCourse courseId={courseId} />} />
          <Route path={`module${num}/exercise/*`} element={<GenericModuleExercise courseId={courseId} />} />
        </React.Fragment>
      ))}
      
      {/* Project and Results */}
      <Route path="project" element={<ProjectPages />} />
      <Route path="project/2024" element={<ProjectPage2024 />} />
      <Route path="project/2025" element={<ProjectPage2025 />} />
      <Route path="project/permuted-mnist" element={<PermutedMNIST />} />
      <Route path="project/bipedal-walker" element={<BipedalWalker />} />
      <Route path="results" element={<RepositoriesList />} />
      <Route path="students/:repoName" element={<StudentsList />} />
      <Route path="student/:repositoryId/:studentId" element={<Student />} />
      <Route path="student-project/:repositoryId/:studentId" element={<StudentProject />} />

      <Route path="*" element={<Navigate to="/courses/data-science-practice" replace />} />
    </Routes>
  );
};

export default DataSciencePractice;