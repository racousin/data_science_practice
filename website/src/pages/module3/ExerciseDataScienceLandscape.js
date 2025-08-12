import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const ExerciseMLPipelineAndExploratoryDataAnalysis = () => {
  const exerciseLinks = [
    {
      to: "/exercise0",
      label: <>Exercise 0<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module3/exercise/Exercise0")),
    },
    {
      to: "/exercise1",
      label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module3/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module3/exercise/Exercise2")),
    },
  ];const module = 3;
  return (
    <ModuleFrame
      module={3}
      isCourse={false}
      title="Module 3"
    >
      
      <Grid>
        <Grid.Col span={{ md: 12 }}>
          <DynamicRoutes routes={exerciseLinks} type="exercise" />
        </Grid.Col>
      </Grid>
    </ModuleFrame>
  );
};
export default ExerciseMLPipelineAndExploratoryDataAnalysis;
