import React, { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useParams, Outlet } from "react-router-dom";
import { MantineProvider, AppShell, Burger, Group, CloseButton, Box } from '@mantine/core';
import '@mantine/core/styles.css';

// Core components
import MainHeader from "./components/MainHeader";
import SideNavigation from "./components/SideNavigation";
import GoogleAnalyticsRouteTracker from "./components/GoogleAnalyticsRouteTracker";

// Main pages
import Home from "./pages/Home";
import CoursesList from "./pages/CoursesList";
import Projects from "./pages/Projects";

// Course containers
import DataSciencePractice from "./courses/DataSciencePractice";
import PythonDeepLearning from "./courses/PythonDeepLearning";

// Initialize Google Analytics
import ReactGA from "react-ga4";
ReactGA.initialize("G-0VW1PJ0H17");

// Layout component for pages without sidebar
const SimpleLayout = () => {
  return (
    <AppShell header={{ height: 60 }} padding="md">
      <AppShell.Header>
        <MainHeader />
      </AppShell.Header>
      <AppShell.Main>
        <Outlet />
      </AppShell.Main>
    </AppShell>
  );
};

// Layout component for pages with sidebar
const CourseLayout = () => {
  const [opened, setOpened] = useState(false);
  
  return (
    <AppShell
      header={{ height: 60 }}
      navbar={{ 
        width: 300, 
        breakpoint: 'md',
        collapsed: { 
          mobile: !opened,
          desktop: false 
        }
      }}
      padding="md"
    >
      <AppShell.Header>
        <MainHeader 
          hamburger={
            <Burger
              opened={opened}
              onClick={() => setOpened((o) => !o)}
              hiddenFrom="md"
              size="sm"
              color="white"
            />
          }
        />
      </AppShell.Header>
      <AppShell.Navbar>
        <SideNavigation onClose={() => setOpened(false)} />
      </AppShell.Navbar>
      <AppShell.Main>
        <Outlet />
      </AppShell.Main>
    </AppShell>
  );
};

function App() {
  return (
    <MantineProvider defaultColorScheme="light">
      <BrowserRouter>
        <GoogleAnalyticsRouteTracker />
        <Routes>
          {/* Routes without sidebar */}
          <Route element={<SimpleLayout />}>
            <Route path="/" element={<Home />} />
            <Route path="/courses" element={<CoursesList />} />
            <Route path="/projects" element={<Projects />} />
          </Route>
          
          {/* Routes with sidebar */}
          <Route element={<CourseLayout />}>
            <Route path="/courses/data-science-practice/*" element={<DataSciencePractice />} />
            <Route path="/courses/python-deep-learning/*" element={<PythonDeepLearning />} />
          </Route>
          
          {/* Legacy redirects */}
          <Route path="/teaching" element={<Navigate to="/courses" replace />} />
          <Route path="/repositories" element={<Navigate to="/courses/data-science-practice/results" replace />} />
          
          {/* Legacy module paths redirect */}
          <Route path="/module:id/*" element={<ModuleRedirect />} />
          <Route path="/project-page" element={<Navigate to="/courses/data-science-practice/project" replace />} />
          
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </MantineProvider>
  );
}

// Helper for legacy URLs
const ModuleRedirect = () => {
  const { id } = useParams();
  return <Navigate to={`/courses/data-science-practice/module${id}`} replace />;
};

export default App;