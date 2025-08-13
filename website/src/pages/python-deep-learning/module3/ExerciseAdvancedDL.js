import React, { lazy } from "react";
import { Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseAdvancedDL = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 3.1: Build Custom MLP",
      component: lazy(() => import("./exercise/Exercise1")),
    },
    {
      to: "/exercise2", 
      label: "Exercise 3.2: Complete Training Pipeline",
      component: lazy(() => import("./exercise/Exercise2")),
    },
    {
      to: "/exercise3", 
      label: "Exercise 3.3: Data & Optimization",
      component: lazy(() => import("./exercise/Exercise3")),
    },
  ];
  
  const module = 3;
  
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 3: Neural Networks & Training Infrastructure - Exercises"
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

export default ExerciseAdvancedDL;