import React, { lazy } from "react";
import { Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseAdvancedDL = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1: Advanced Architectures",
      component: lazy(() => import("./exercise/Exercise1")),
    },
    {
      to: "/exercise2", 
      label: "Exercise 2: Specialized Networks",
      component: lazy(() => import("./exercise/Exercise2")),
    },
  ];
  
  const module = 3;
  
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 3: Advanced Deep Learning Exercises"
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