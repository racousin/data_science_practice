import React, { useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate, Link, useLocation } from "react-router-dom";
import { MantineProvider, AppShell, Group, Text, Container, Button } from '@mantine/core';
import ReactGA from "react-ga4";

import Home from "./pages/Home";
import RepositoriesList from "pages/RepositoriesList";
import StudentsList from "./pages/StudentsList";
import Student from "./pages/Student";
import Teaching from "./pages/Teaching";

import PrerequistAandMethodologie from "pages/module0/PrerequistAandMethodologie";
import CourseGit from "pages/module1/CourseGit";
import ExerciseGit from "pages/module1/ExerciseGit";
import CoursePython from "pages/module2/CoursePython";
import ExercisePython from "pages/module2/ExercisePython";
import CourseDataScienceLandscape from "pages/module3/CourseDataScienceLandscape";
import ExerciseMLPipelineAndExploratoryDataAnalysis from "pages/module3/ExerciseDataScienceLandscape";
import CourseDataCollection from "pages/module4/CourseDataCollection";
import ExerciseDataCollection from "pages/module4/ExerciseDataCollection";
import CourseDataCleaningAndPreparation from "pages/module5/CourseDataCleaningAndPreparation";
import ExerciseDataCleaningAndPreparation from "pages/module5/ExerciseDataCleaningAndPreparation";
import CourseTabularModels from "pages/module6/CourseTabularModels";
import ExerciseTabularModels from "pages/module6/ExerciseTabularModels";
import CourseDeepLearningFundamentals from "pages/module7/CourseDeepLearningFundamentals";
import ExerciseDeepLearningFundamentals from "pages/module7/ExerciseDeepLearningFundamentals";
import CourseImageProcessing from "pages/module8/CourseImageProcessing";
import ExerciseImageProcessing from "pages/module8/ExerciseImageProcessing";
import CourseTextProcessing from "pages/module10/CourseTextProcessing";
import ExerciseTextProcessing from "pages/module10/ExerciseTextProcessing";
import CourseGenerativeModels from "pages/module11/CourseGenerativeModels";
import ExerciseGenerativeModels from "pages/module11/ExerciseGenerativeModels";
import CourseRecommendationSystems from "pages/module12/CourseRecommendationSystems";
import ExerciseRecommendationSystems from "pages/module12/ExerciseRecommendationSystems";
import CourseReinforcementLearning from "pages/module13/CourseReinforcementLearning";
import ExerciseReinforcementLearning from "pages/module13/ExerciseReinforcementLearning";
import CourseDocker from "pages/module14/CourseDocker";
import ExerciseDocker from "pages/module14/ExerciseDocker";
import CourseCloudIntegration from "pages/module15/CourseCloudIntegration";
import ExerciseCloudIntegration from "pages/module15/ExerciseCloudIntegration";

import ApiDoc from './pages/module4/ApiDoc';
import ScrapableData from './pages/module4/ScrapableData';

import PageToScrap from "pages/module4/course/PageToScrap";
import PageToScrapExercise from "pages/module4/course/PageToScrapExercise";

import Resources from "pages/Resources";

import "App.css";

import SearchResultsPage from "pages/SearchResultsPage";

ReactGA.initialize("G-0VW1PJ0H17"); 

const RobotsTxt = () => {
  // Sitemap: https://example.com/sitemap.xml TODO
  return (
    <pre>
      {`User-agent: *
Allow: /
Disallow: /private/
Disallow: /admin/

User-agent: Googlebot
Allow: /

Crawl-delay: 5


`}
    </pre>
  );
};

function App() {
  return (
    <MantineProvider withGlobalStyles withNormalizeCSS>
      <BrowserRouter>
        <AppShell
          header={{ height: 60 }}
          padding="md"
        >
        <AppShell.Header style={{ backgroundColor: 'black' }}>
          <Container size="xl" h="100%">
            <Group justify="space-between" align="center" h="100%">
              <Text 
                component={Link} 
                to="/" 
                size="xl" 
                fw={700} 
                style={{ textDecoration: 'none', color: 'white' }}
              >
                Teaching Portal
              </Text>
              <Group>
                <Button 
                  component={Link} 
                  to="/" 
                  variant="subtle" 
                  styles={{ root: { color: 'white' } }}
                >
                  Home
                </Button>
                <Button 
                  component={Link} 
                  to="/teaching" 
                  variant="subtle" 
                  styles={{ root: { color: 'white' } }}
                >
                  Teaching
                </Button>
                <Button 
                  component={Link} 
                  to="/repositories" 
                  variant="subtle" 
                  styles={{ root: { color: 'white' } }}
                >
                  Sessions Results
                </Button>
              </Group>
            </Group>
          </Container>
        </AppShell.Header>


          <AppShell.Main>
          <Container fluid>
              <GoogleAnalyticsRouteTracker />
              <Routes>
        
          <Route path="/" element={<Home />} />
          <Route path="repositories" element={<RepositoriesList />} />
          <Route path="students/:repositoryId" element={<StudentsList />} />
          <Route
            path="student/:repositoryId/:studentId"
            element={<Student />}
          />
          <Route path="teaching" element={<Teaching />} />
          <Route path="resources" element={<Resources />} />
          <Route path="robots.txt" element={<RobotsTxt />} />
          <Route
            path="module0/course"
            element={<PrerequistAandMethodologie />}
          />
          <Route path="module1/course/*" element={<CourseGit />} />
          <Route path="module1/exercise/*" element={<ExerciseGit />} />
          <Route path="module2/course/*" element={<CoursePython />} />
          <Route path="module2/exercise/*" element={<ExercisePython />} />
          <Route
            path="module3/course/*"
            element={<CourseDataScienceLandscape />}
          />
          <Route
            path="module3/exercise/*"
            element={<ExerciseMLPipelineAndExploratoryDataAnalysis />}
          />
          <Route path="module4/course/*" element={<CourseDataCollection />} />
          <Route path="module4/api-doc" element={<ApiDoc />} />
          <Route path="module4/scrapable-data" element={<ScrapableData />} />
          <Route
            path="module4/exercise/*"
            element={<ExerciseDataCollection />}
          />
          <Route path="page-to-scrap" element={<PageToScrap />} />
          <Route
            path="page-to-scrap-exercise/*"
            element={<PageToScrapExercise />}
          />
          <Route
            path="module5/course/*"
            element={<CourseDataCleaningAndPreparation />}
          />
          <Route
            path="module5/exercise/*"
            element={<ExerciseDataCleaningAndPreparation />}
          />
          <Route path="module6/course/*" element={<CourseTabularModels />} />
          <Route
            path="module6/exercise/*"
            element={<ExerciseTabularModels />}
          />
          <Route
            path="module7/course/*"
            element={<CourseDeepLearningFundamentals />}
          />
          <Route
            path="module7/exercise/*"
            element={<ExerciseDeepLearningFundamentals />}
          />
          <Route path="module8/course/*" element={<CourseImageProcessing />} />
          <Route
            path="module8/exercise/*"
            element={<ExerciseImageProcessing />}
          />

          <Route path="module10/course/*" element={<CourseTextProcessing />} />
          <Route
            path="module10/exercise/*"
            element={<ExerciseTextProcessing />}
          />
          <Route
            path="module11/course/*"
            element={<CourseGenerativeModels />}
          />
          <Route
            path="module11/exercise/*"
            element={<ExerciseGenerativeModels />}
          />
          <Route
            path="module12/course/*"
            element={<CourseRecommendationSystems />}
          />
          <Route
            path="module12/exercise/*"
            element={<ExerciseRecommendationSystems />}
          />
          <Route
            path="module13/course/*"
            element={<CourseReinforcementLearning />}
          />
          <Route
            path="module13/exercise/*"
            element={<ExerciseReinforcementLearning />}
          />
          <Route path="module14/course/*" element={<CourseDocker />} />
          <Route path="module14/exercise/*" element={<ExerciseDocker />} />
          <Route
            path="module15/course/*"
            element={<CourseCloudIntegration />}
          />
          <Route
            path="module15/exercise/*"
            element={<CourseCloudIntegration />}
          />

          <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
            </Container>
          </AppShell.Main>
        </AppShell>
      </BrowserRouter>
    </MantineProvider>
  );
}

function GoogleAnalyticsRouteTracker() {
  const location = useLocation();

  useEffect(() => {
    ReactGA.send({ hitType: "pageview", page: location.pathname });
  }, [location]);

  return null;
}

export default App;
