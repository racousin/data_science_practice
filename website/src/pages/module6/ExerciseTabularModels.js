import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const ExerciseTabularModels = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module6/exercise/Exercise1")),
    },
    // Add links to other exercises as needed
  ];const module = 6;
  return (
    <ModuleFrame
      module={6}
      isCourse={false}
      title="Module 6: Exercise Building and Evaluation"
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
export default ExerciseTabularModels;
