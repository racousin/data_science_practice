import React, { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useParams, Outlet } from "react-router-dom";
import { MantineProvider, AppShell, Burger } from '@mantine/core';
import '@mantine/core/styles.css';

// Core components
import MainHeader from "./components/MainHeader";
import SideNavigation from "./components/SideNavigation";
import GoogleAnalyticsRouteTracker from "./components/GoogleAnalyticsRouteTracker";
import { SidebarProvider, useSidebar } from "./contexts/SidebarContext";

// Main pages
import Home from "./pages/Home";
import CoursesList from "./pages/CoursesList";
import Projects from "./pages/Projects";

// Course containers
import DataSciencePractice from "./courses/DataSciencePractice";
import PythonDeepLearning from "./courses/PythonDeepLearning";

// Standalone data endpoints
import ScrapableData from "./pages/data-science-practice/module4/ScrapableData";
import ApiDoc from "./pages/data-science-practice/module4/ApiDoc";

// Project pages
import PermutedMNIST from "./pages/data-science-practice/project-pages/PermutedMNIST";
import BipedalWalker from "./pages/data-science-practice/project-pages/BipedalWalker";

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
    <SidebarProvider 
      sidebarOpened={opened} 
      toggleSidebar={() => setOpened((o) => !o)}
    >
      <CourseLayoutInner opened={opened} setOpened={setOpened} />
    </SidebarProvider>
  );
};

// Inner component to access slide mode context
const CourseLayoutInner = ({ opened, setOpened }) => {
  const { slideMode } = useSidebar();
  
  return (
    <AppShell
      header={slideMode ? undefined : { height: 60 }}
      navbar={slideMode ? undefined : { 
        width: 300, 
        breakpoint: 'md',
        collapsed: { 
          mobile: !opened,
          desktop: opened 
        }
      }}
      padding={slideMode ? 0 : "md"}
    >
      {!slideMode && (
        <AppShell.Header>
          <MainHeader 
            hamburger={
              <Burger
                opened={opened}
                onClick={() => setOpened((o) => !o)}
                size="sm"
                color="white"
              />
            }
          />
        </AppShell.Header>
      )}
      {!slideMode && (
        <AppShell.Navbar>
          <SideNavigation onClose={() => setOpened(false)} />
        </AppShell.Navbar>
      )}
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
            <Route path="/module4/scrapable-data" element={<ScrapableData />} />
            <Route path="/module4/api-doc" element={<ApiDoc />} />
            <Route path="/data-science-practice/project-pages/permuted-mnist" element={<PermutedMNIST />} />
            <Route path="/data-science-practice/project-pages/bipedal-walker" element={<BipedalWalker />} />
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