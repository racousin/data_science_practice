import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Course overview page
import CourseOverview from './PythonDeepLearningOverview';

// Module pages
import CourseTensors from '../pages/python-deep-learning/module1/CourseTensors';
import CourseNeuralNetworks from '../pages/python-deep-learning/module2/CourseNeuralNetworks';
import CourseAdvancedDL from '../pages/python-deep-learning/module3/CourseAdvancedDL';
import CourseProduction from '../pages/python-deep-learning/module4/CourseProduction';

const PythonDeepLearning = () => {
  return (
    <Routes>
      <Route path="/" element={<CourseOverview />} />
      
      {/* Module 1 - PyTorch Fundamentals */}
      <Route path="module1" element={<CourseTensors />} />
      <Route path="module1/course/*" element={<CourseTensors />} />
      
      {/* Module 2 - Neural Network Architectures */}
      <Route path="module2" element={<CourseNeuralNetworks />} />
      <Route path="module2/course/*" element={<CourseNeuralNetworks />} />
      
      {/* Module 3 - Advanced Deep Learning */}
      <Route path="module3" element={<CourseAdvancedDL />} />
      <Route path="module3/course/*" element={<CourseAdvancedDL />} />
      
      {/* Module 4 - Production and Deployment */}
      <Route path="module4" element={<CourseProduction />} />
      <Route path="module4/course/*" element={<CourseProduction />} />
      
      <Route path="*" element={<Navigate to="/courses/python-deep-learning" replace />} />
    </Routes>
  );
};

export default PythonDeepLearning;