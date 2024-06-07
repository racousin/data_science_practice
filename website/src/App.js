import React from "react";
import { BrowserRouter, Routes, Route, Navigate, Link } from "react-router-dom";
import { Navbar, Nav, Container } from "react-bootstrap";
import Home from "./pages/Home";
import StudentsList from "./pages/StudentsList";
import Student from "./pages/Student";
import Teaching from "./pages/Teaching";
import CourseGit from "pages/module1/CourseGit";
import ExerciseEvaluation from "pages/ExerciseEvaluation";
import ExerciseGit from "pages/module1/ExerciseGit";
import CoursePython from "pages/module2/CoursePython";
import ExercisePython from "pages/module2/ExercisePython";
import CourseDataScienceOverview from "pages/module3/CourseDataScienceOverview";
import ExerciseDataScienceOverview from "pages/module3/ExerciseDataScienceOverview";
import CourseFeatureEngineering from "pages/module5/CourseFeatureEngineering";
import ExerciseFeatureEngineering from "pages/module5/ExerciseFeatureEngineering";
import CourseGetData from "pages/module4/CourseGetData";
import ExerciseGetData from "pages/module4/ExerciseGetData";
import CourseModelBuildingEvaluation from "pages/module6/CourseModelBuildingEvaluation";
import ExerciseModelBuildingEvaluation from "pages/module6/ExerciseModelBuildingEvaluation";
import CourseAdvancedTabularModels from "pages/module7/CourseAdvancedTabularModels";
import ExerciseAdvancedTabularModels from "pages/module7/ExerciseAdvancedTabularModels";
import CourseDeepLearningFundamentals from "pages/module8/CourseDeepLearningFundamentals";
import ExerciseDeepLearningFundamentals from "pages/module8/ExerciseDeepLearningFundamentals";
import CourseDocker from "pages/module9/CourseDocker";
import ExerciseDocker from "pages/module9/ExerciseDocker";
import CourseCloudIntegration from "pages/module10/CourseCloudIntegration";
import ExerciseCloudIntegration from "pages/module10/ExerciseCloudIntegration";
import CourseTextProcessing from "pages/module12/CourseTextProcessing";
import ExerciseTextProcessing from "pages/module12/ExerciseTextProcessing";
import CourseRecommendationSystems from "pages/module13/CourseRecommendationSystems";
import ExerciseRecommendationSystems from "pages/module13/ExerciseRecommendationSystems";
import CourseGenerativeModels from "pages/module14/CourseGenerativeModels";
import ExerciseGenerativeModels from "pages/module14/ExerciseGenerativeModels";
import CourseReinforcementLearning from "pages/module15/CourseReinforcementLearning";
import ExerciseReinforcementLearning from "pages/module15/ExerciseReinforcementLearning";

import "App.css";
import CourseImageProcessing from "pages/module11/CourseImageProcessing";
import ExerciseImageProcessing from "pages/module11/ExerciseImageProcessing";

function App() {
  return (
    <BrowserRouter>
      <Navbar bg="dark" variant="dark" expand="lg" sticky="top">
        <Container>
          <Navbar.Brand as={Link} to="/">
            Teaching Portal
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link as={Link} to="/">
                Home
              </Nav.Link>
              <Nav.Link as={Link} to="/teaching">
                Teaching
              </Nav.Link>

              <Nav.Link as={Link} to="/students">
                Student
              </Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>
      <Container className="mt-3">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="students" element={<StudentsList />} />
          <Route path="student/:studentId" element={<Student />} />
          <Route path="teaching" element={<Teaching />} />
          <Route path="module1/course/*" element={<CourseGit />} />
          <Route path="module1/exercise/*" element={<ExerciseGit />} />
          <Route path="module2/course/*" element={<CoursePython />} />
          <Route path="module2/exercise/*" element={<ExercisePython />} />
          <Route
            path="module3/course/*"
            element={<CourseDataScienceOverview />}
          />
          <Route
            path="module3/exercise/*"
            element={<ExerciseDataScienceOverview />}
          />
          <Route path="module4/course/*" element={<CourseGetData />} />
          <Route path="module4/exercise/*" element={<ExerciseGetData />} />
          <Route
            path="module5/course/*"
            element={<CourseFeatureEngineering />}
          />
          <Route
            path="module5/exercise/*"
            element={<ExerciseFeatureEngineering />}
          />
          <Route
            path="module6/course/*"
            element={<CourseModelBuildingEvaluation />}
          />
          <Route
            path="module6/exercise/*"
            element={<ExerciseModelBuildingEvaluation />}
          />
          <Route
            path="module7/course/*"
            element={<CourseAdvancedTabularModels />}
          />
          <Route
            path="module7/exercise/*"
            element={<ExerciseAdvancedTabularModels />}
          />
          <Route
            path="module8/course/*"
            element={<CourseDeepLearningFundamentals />}
          />
          <Route
            path="module8/exercise/*"
            element={<ExerciseDeepLearningFundamentals />}
          />
          <Route path="module9/course/*" element={<CourseDocker />} />
          <Route path="module9/exercise/*" element={<ExerciseDocker />} />

          <Route
            path="module10/course/*"
            element={<CourseCloudIntegration />}
          />
          <Route
            path="module10/exercise/*"
            element={<ExerciseCloudIntegration />}
          />
          <Route path="module11/course/*" element={<CourseImageProcessing />} />
          <Route
            path="module11/exercise/*"
            element={<ExerciseImageProcessing />}
          />
          <Route path="module12/course/*" element={<CourseTextProcessing />} />
          <Route
            path="module12/exercise/*"
            element={<ExerciseTextProcessing />}
          />
          <Route
            path="module13/course/*"
            element={<CourseRecommendationSystems />}
          />
          <Route
            path="module13/exercise/*"
            element={<ExerciseRecommendationSystems />}
          />
          <Route
            path="module14/course/*"
            element={<CourseGenerativeModels />}
          />
          <Route
            path="module14/exercise/*"
            element={<ExerciseGenerativeModels />}
          />
          <Route
            path="module15/course/*"
            element={<CourseReinforcementLearning />}
          />
          <Route
            path="module15/exercise/*"
            element={<ExerciseReinforcementLearning />}
          />
          <Route path="exercise-evaluation" element={<ExerciseEvaluation />} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Container>
    </BrowserRouter>
  );
}

export default App;
