import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const ExerciseImageProcessing = () => {
  const exerciseLinks = [
    {
      to: "/exercise0",
      label: "Exercise 0",
      component: lazy(() => import("pages/module8/exercise/Exercise0")),
    },
    {
      to: "/exercise1",
      label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module8/exercise/Exercise1")),
    },
    // Add links to other exercises as needed
  ];const module = 8;
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 8: Exercise Image Processing"
      courseLinks={exerciseLinks}
    >
      
      <Grid>
        <Grid.Col span={{ md: 11 }}>
          <DynamicRoutes routes={exerciseLinks} type="exercise" />
        </Grid.Col>
      </Grid>
    </ModuleFrame>
  );
};
export default ExerciseImageProcessing;
