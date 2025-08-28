import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Course overview page
import CourseOverview from './PythonDeepLearningOverview';

// Generic module components
import GenericModuleCourse from '../components/GenericModuleCourse';
import GenericModuleExercise from '../components/GenericModuleExercise';

const PythonDeepLearning = () => {
  const courseId = 'python-deep-learning';
  
  return (
    <Routes>
      <Route path="/" element={<CourseOverview />} />
      
      {/* Module 1 - Foundations of Deep Learning */}
      <Route path="module1" element={<GenericModuleCourse courseId={courseId} />} />
      <Route path="module1/course/*" element={<GenericModuleCourse courseId={courseId} />} />
      <Route path="module1/exercise/*" element={<GenericModuleExercise courseId={courseId} />} />
      
      {/* Module 2 - Automatic Differentiation & Optimization */}
      <Route path="module2" element={<GenericModuleCourse courseId={courseId} />} />
      <Route path="module2/course/*" element={<GenericModuleCourse courseId={courseId} />} />
      <Route path="module2/exercise/*" element={<GenericModuleExercise courseId={courseId} />} />
      
      {/* Module 3 - Neural Networks & Training Infrastructure */}
      <Route path="module3" element={<GenericModuleCourse courseId={courseId} />} />
      <Route path="module3/course/*" element={<GenericModuleCourse courseId={courseId} />} />
      <Route path="module3/exercise/*" element={<GenericModuleExercise courseId={courseId} />} />
      
      {/* Module 4 - Performance Optimization & Advanced Features */}
      <Route path="module4" element={<GenericModuleCourse courseId={courseId} />} />
      <Route path="module4/course/*" element={<GenericModuleCourse courseId={courseId} />} />
      <Route path="module4/exercise/*" element={<GenericModuleExercise courseId={courseId} />} />
      
      <Route path="*" element={<Navigate to="/courses/python-deep-learning" replace />} />
    </Routes>
  );
};

export default PythonDeepLearning;