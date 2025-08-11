import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Container } from '@mantine/core';

// Module pages
import PrerequistAndMethodologie from '../pages/module0/PrerequistAandMethodologie';
import CourseGit from '../pages/module1/CourseGit';
import ExerciseGit from '../pages/module1/ExerciseGit';
import CoursePython from '../pages/module2/CoursePython';
import ExercisePython from '../pages/module2/ExercisePython';
import CourseDataScienceLandscape from '../pages/module3/CourseDataScienceLandscape';
import ExerciseDataScienceLandscape from '../pages/module3/ExerciseDataScienceLandscape';
import CourseDataCollection from '../pages/module4/CourseDataCollection';
import ExerciseDataCollection from '../pages/module4/ExerciseDataCollection';
import CourseDataPreprocessing from '../pages/module5/CourseDataPreprocessing';
import ExerciseDataPreprocessing from '../pages/module5/ExerciseDataPreprocessing';
import CourseTabularModels from '../pages/module6/CourseTabularModels';
import ExerciseTabularModels from '../pages/module6/ExerciseTabularModels';
import CourseDeepLearningFundamentals from '../pages/module7/CourseDeepLearningFundamentals';
import ExerciseDeepLearningFundamentals from '../pages/module7/ExerciseDeepLearningFundamentals';
import CourseImageProcessing from '../pages/module8/CourseImageProcessing';
import ExerciseImageProcessing from '../pages/module8/ExerciseImageProcessing';
import CourseTimeSeriesProcessing from '../pages/module9/CourseTimeSeriesProcessing';
import ExerciseTimeSeriesProcessing from '../pages/module9/ExerciseTimeSeriesProcessing';
import CourseTextProcessing from '../pages/module10/CourseTextProcessing';
import ExerciseTextProcessing from '../pages/module10/ExerciseTextProcessing';
import CourseGenerativeModels from '../pages/module11/CourseGenerativeModels';
import ExerciseGenerativeModels from '../pages/module11/ExerciseGenerativeModels';
import CourseRecommendationSystems from '../pages/module12/CourseRecommendationSystems';
import ExerciseRecommendationSystems from '../pages/module12/ExerciseRecommendationSystems';
import CourseReinforcementLearning from '../pages/module13/CourseReinforcementLearning';
import ExerciseReinforcementLearning from '../pages/module13/ExerciseReinforcementLearning';
import CourseDocker from '../pages/module14/CourseDocker';
import ExerciseDocker from '../pages/module14/ExerciseDocker';
import CourseCloudIntegration from '../pages/module15/CourseCloudIntegration';
import ExerciseCloudIntegration from '../pages/module15/ExerciseCloudIntegration';
import ProjectPage from '../pages/ProjectPage';
import RepositoriesList from '../pages/RepositoriesList';
import StudentsList from '../pages/StudentsList';

// Course overview page
import CourseOverview from './DataSciencePracticeOverview';

const DataSciencePractice = () => {
  return (
    <Routes>
      <Route path="/" element={<CourseOverview />} />
      
      {/* Module 0 */}
      <Route path="module0" element={<PrerequistAndMethodologie />} />
      
      {/* Module 1 - Git */}
      <Route path="module1" element={<CourseGit />} />
      <Route path="module1/course/*" element={<CourseGit />} />
      <Route path="module1/exercise/*" element={<ExerciseGit />} />
      
      {/* Module 2 - Python */}
      <Route path="module2" element={<CoursePython />} />
      <Route path="module2/course/*" element={<CoursePython />} />
      <Route path="module2/exercise/*" element={<ExercisePython />} />
      
      {/* Module 3 - Data Science Landscape */}
      <Route path="module3" element={<CourseDataScienceLandscape />} />
      <Route path="module3/course/*" element={<CourseDataScienceLandscape />} />
      <Route path="module3/exercise/*" element={<ExerciseDataScienceLandscape />} />
      
      {/* Module 4 - Data Collection */}
      <Route path="module4" element={<CourseDataCollection />} />
      <Route path="module4/course/*" element={<CourseDataCollection />} />
      <Route path="module4/exercise/*" element={<ExerciseDataCollection />} />
      
      {/* Module 5 - Data Preprocessing */}
      <Route path="module5" element={<CourseDataPreprocessing />} />
      <Route path="module5/course/*" element={<CourseDataPreprocessing />} />
      <Route path="module5/exercise/*" element={<ExerciseDataPreprocessing />} />
      
      {/* Module 6 - Tabular Models */}
      <Route path="module6" element={<CourseTabularModels />} />
      <Route path="module6/course/*" element={<CourseTabularModels />} />
      <Route path="module6/exercise/*" element={<ExerciseTabularModels />} />
      
      {/* Module 7 - Deep Learning Fundamentals */}
      <Route path="module7" element={<CourseDeepLearningFundamentals />} />
      <Route path="module7/course/*" element={<CourseDeepLearningFundamentals />} />
      <Route path="module7/exercise/*" element={<ExerciseDeepLearningFundamentals />} />
      
      {/* Module 8 - Image Processing */}
      <Route path="module8" element={<CourseImageProcessing />} />
      <Route path="module8/course/*" element={<CourseImageProcessing />} />
      <Route path="module8/exercise/*" element={<ExerciseImageProcessing />} />
      
      {/* Module 9 - Time Series Processing */}
      <Route path="module9" element={<CourseTimeSeriesProcessing />} />
      <Route path="module9/course/*" element={<CourseTimeSeriesProcessing />} />
      <Route path="module9/exercise/*" element={<ExerciseTimeSeriesProcessing />} />
      
      {/* Module 10 - Text Processing */}
      <Route path="module10" element={<CourseTextProcessing />} />
      <Route path="module10/course/*" element={<CourseTextProcessing />} />
      <Route path="module10/exercise/*" element={<ExerciseTextProcessing />} />
      
      {/* Module 11 - Generative Models */}
      <Route path="module11" element={<CourseGenerativeModels />} />
      <Route path="module11/course/*" element={<CourseGenerativeModels />} />
      <Route path="module11/exercise/*" element={<ExerciseGenerativeModels />} />
      
      {/* Module 12 - Recommendation Systems */}
      <Route path="module12" element={<CourseRecommendationSystems />} />
      <Route path="module12/course/*" element={<CourseRecommendationSystems />} />
      <Route path="module12/exercise/*" element={<ExerciseRecommendationSystems />} />
      
      {/* Module 13 - Reinforcement Learning */}
      <Route path="module13" element={<CourseReinforcementLearning />} />
      <Route path="module13/course/*" element={<CourseReinforcementLearning />} />
      <Route path="module13/exercise/*" element={<ExerciseReinforcementLearning />} />
      
      {/* Module 14 - Docker */}
      <Route path="module14" element={<CourseDocker />} />
      <Route path="module14/course/*" element={<CourseDocker />} />
      <Route path="module14/exercise/*" element={<ExerciseDocker />} />
      
      {/* Module 15 - Cloud Integration */}
      <Route path="module15" element={<CourseCloudIntegration />} />
      <Route path="module15/course/*" element={<CourseCloudIntegration />} />
      <Route path="module15/exercise/*" element={<ExerciseCloudIntegration />} />
      
      {/* Project and Results */}
      <Route path="project" element={<ProjectPage />} />
      <Route path="results" element={<RepositoriesList />} />
      <Route path="students/:repoName" element={<StudentsList />} />
      
      <Route path="*" element={<Navigate to="/courses/data-science-practice" replace />} />
    </Routes>
  );
};

export default DataSciencePractice;