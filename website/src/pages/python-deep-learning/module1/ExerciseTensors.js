import React, { lazy } from "react";
import { Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseTensors = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1.1: Environment & Basics",
      component: lazy(() => import("./exercise/Exercise1")),
    },
    {
      to: "/exercise2", 
      label: "Exercise 1.2: Mathematical Implementation",
      component: lazy(() => import("./exercise/Exercise2")),
    },
    {
      to: "/exercise3", 
      label: "Exercise 1.3: Tensor Mastery",
      component: lazy(() => import("./exercise/Exercise3")),
    },
  ];
  
  const module = 1;
  
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 1: Foundations & Mathematical Framework - Exercises"
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