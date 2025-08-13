import React, { lazy } from "react";
import { Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseProduction = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 4.1: Performance Profiling",
      component: lazy(() => import("./exercise/Exercise1")),
    },
    {
      to: "/exercise2", 
      label: "Exercise 4.2: Advanced Features",
      component: lazy(() => import("./exercise/Exercise2")),
    },
    {
      to: "/exercise3", 
      label: "Exercise 4.3: Mini-Project",
      component: lazy(() => import("./exercise/Exercise3")),
    },
  ];
  
  const module = 4;
  
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 4: Performance Optimization & Advanced Features - Exercises"
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

export default ExerciseProduction;