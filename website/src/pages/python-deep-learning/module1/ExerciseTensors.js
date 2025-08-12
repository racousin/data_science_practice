import React, { lazy } from "react";
import { Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseTensors = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1: Tensor Basics",
      component: lazy(() => import("./exercise/Exercise1")),
    },
    {
      to: "/exercise2", 
      label: "Exercise 2: PyTorch Operations",
      component: lazy(() => import("./exercise/Exercise2")),
    },
  ];
  
  const module = 1;
  
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 1: PyTorch Fundamentals Exercises"
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

export default ExerciseTensors;