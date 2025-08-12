import React, { Suspense, lazy } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import ScrollToTop from "components/ScrollToTop";
import ModuleTableOfContents from "./ModuleTableOfContents";

const DynamicRoutes = ({ routes, type = 'course' }) => {
  const location = useLocation();
  const pathParts = location.pathname.split('/').filter(Boolean);
  
  // Check if we're on the overview page (no specific route selected)
  const isOverviewPage = pathParts.length === 4 && 
    (pathParts[3] === 'course' || pathParts[3] === 'exercise');
  
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ScrollToTop />
      {isOverviewPage ? (
        <ModuleTableOfContents type={pathParts[3] === 'exercise' ? 'exercise' : 'course'} />
      ) : (
        <Routes>
          {routes.map(({ to, component: Component }) => (
            <Route key={to} path={to} element={<Component />} />
          ))}
        </Routes>
      )}
    </Suspense>
  );
};

export default DynamicRoutes;
