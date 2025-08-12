import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Course overview page
import CourseOverview from './PythonDeepLearningOverview';

// Module pages
import CourseTensors from '../pages/python-deep-learning/module1/CourseTensors';

const Module2 = () => (
  <div style={{ padding: '20px' }}>
    <h1>Module 2: PyTorch Fundamentals</h1>
    <p>This module is under construction.</p>
  </div>
);

const Module3 = () => (
  <div style={{ padding: '20px' }}>
    <h1>Module 3: TensorBoard Visualization</h1>
    <p>This module is under construction.</p>
  </div>
);

const PythonDeepLearning = () => {
  return (
    <Routes>
      <Route path="/" element={<CourseOverview />} />
      
      {/* Module 1 - Tensors */}
      <Route path="module1" element={<CourseTensors />} />
      <Route path="module1/course/*" element={<CourseTensors />} />
      
      {/* Module 2 - PyTorch */}
      <Route path="module2" element={<Module2 />} />
      <Route path="module2/*" element={<Module2 />} />
      
      {/* Module 3 - TensorBoard */}
      <Route path="module3" element={<Module3 />} />
      <Route path="module3/*" element={<Module3 />} />
      
      <Route path="*" element={<Navigate to="/courses/python-deep-learning" replace />} />
    </Routes>
  );
};

export default PythonDeepLearning;