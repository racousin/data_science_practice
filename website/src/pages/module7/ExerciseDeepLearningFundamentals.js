import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const ExerciseDeepLearningFundamentals = () => {
  const exerciseLinks = [
    {
      to: "/exercise0",
      label: "Exercise 0",
      component: lazy(() => import("pages/module7/exercise/Exercise0")),
    },
    {
      to: "/exercise1",
      label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module7/exercise/Exercise1")),
    },
    // Add links to other exercises as needed
  ];
  const location = useLocation();
  const module = 7;
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 8: Exercise Deep Learning Fundamentals"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Last Updated: {"2024-11-18"}</p>
            </Grid.Col>
          </Grid>
        </>
      )}
      <Grid>
        <Grid.Col span={{ md: 11 }}>
          <DynamicRoutes routes={exerciseLinks} />
        </Grid.Col>
      </Grid>
    </ModuleFrame>
  );
};
export default ExerciseDeepLearningFundamentals;
