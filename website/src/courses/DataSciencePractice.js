import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Generic module components
import GenericModuleCourse from '../components/GenericModuleCourse';
import GenericModuleExercise from '../components/GenericModuleExercise';

// Special pages
import PrerequistAndMethodologie from '../pages/data-science-practice/module0/PrerequistAandMethodologie';
import ProjectPage from '../pages/ProjectPage';
import RepositoriesList from '../pages/RepositoriesList';
import StudentsList from '../pages/StudentsList';

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
      <Route path="project" element={<ProjectPage />} />
      <Route path="results" element={<RepositoriesList />} />
      <Route path="students/:repoName" element={<StudentsList />} />
      
      <Route path="*" element={<Navigate to="/courses/data-science-practice" replace />} />
    </Routes>
  );
};

export default DataSciencePractice;