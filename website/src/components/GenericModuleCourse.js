import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation, useParams } from "react-router-dom";
import { getCourseContentLinks, coursesData } from './SideNavigation';

const GenericModuleCourse = ({ courseId }) => {
  const location = useLocation();
  const params = useParams();
  
  // Extract module number from URL
  const moduleMatch = location.pathname.match(/module(\d+)/);
  const moduleNum = moduleMatch ? parseInt(moduleMatch[1]) : 1;
  const moduleId = `module${moduleNum}`;
  
  // Get course links from SideNavigation
  const courseLinks = getCourseContentLinks(moduleId, courseId);
  
  // Transform links to include lazy-loaded components
  const routeLinks = courseLinks.map(link => {
    const basePath = courseId === 'python-deep-learning' 
      ? `pages/python-deep-learning/${moduleId}/course`
      : `pages/data-science-practice/${moduleId}/course`;
    
    // Convert path to component name (e.g., '/introduction' -> 'Introduction')
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
      isCourse={true}
      title={`Module ${moduleNum}: ${moduleName}`}
      courseLinks={routeLinks}
      enableSlides={courseId === 'python-deep-learning'}
    >
      {location.pathname === `/courses/${courseId}/${moduleId}/course` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Last Updated: {new Date().toISOString().split('T')[0]}</p>
            </Grid.Col>
          </Grid>
        </>
      )}
      <Grid>
        <Grid.Col span={{ md: 11 }}>
          <DynamicRoutes routes={routeLinks} type="course" />
        </Grid.Col>
      </Grid>
    </ModuleFrame>
  );
};

export default GenericModuleCourse;