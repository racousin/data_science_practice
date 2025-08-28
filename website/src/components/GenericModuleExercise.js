import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
import { getExerciseContentLinks, coursesData } from './SideNavigation';

const GenericModuleExercise = ({ courseId }) => {
  const location = useLocation();
  
  // Extract module number from URL
  const moduleMatch = location.pathname.match(/module(\d+)/);
  const moduleNum = moduleMatch ? parseInt(moduleMatch[1]) : 1;
  const moduleId = `module${moduleNum}`;
  
  // Get exercise links from SideNavigation
  const exerciseLinks = getExerciseContentLinks(moduleId, courseId);
  
  // Transform links to include lazy-loaded components
  const routeLinks = exerciseLinks.map(link => {
    const basePath = courseId === 'python-deep-learning' 
      ? `pages/python-deep-learning/${moduleId}/exercise`
      : `pages/data-science-practice/${moduleId}/exercise`;
    
    // Convert path to component name (e.g., '/exercise1' -> 'Exercise1')
    const componentName = link.to.slice(1).split('-').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join('');
    
    return {
      ...link,
      component: lazy(() => import(`../${basePath}/${componentName}`))
    };
  });
  
  // Get module info
  const courseInfo = coursesData[courseId];
  const moduleInfo = courseInfo?.modules.find(m => m.id === moduleId);
  const moduleName = moduleInfo?.name || `Module ${moduleNum}`;
  
  return (
    <ModuleFrame
      module={moduleNum}
      isCourse={false}
      title={`Module ${moduleNum}: ${moduleName} - Exercises`}
      courseLinks={routeLinks}
    >
      {location.pathname === `/courses/${courseId}/${moduleId}/exercise` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Complete the exercises below to practice the concepts from this module.</p>
            </Grid.Col>
          </Grid>
        </>
      )}
      <Grid>
        <Grid.Col span={{ md: 11 }}>
          <DynamicRoutes routes={routeLinks} type="exercise" />
        </Grid.Col>
      </Grid>
    </ModuleFrame>
  );
};

export default GenericModuleExercise;