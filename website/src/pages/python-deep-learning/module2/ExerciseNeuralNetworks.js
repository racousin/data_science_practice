import React, { lazy } from "react";
import { Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const ExerciseNeuralNetworks = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: "Exercise 1: Building Neural Networks",
      component: lazy(() => import("./exercise/Exercise1")),
    },
    {
      to: "/exercise2", 
      label: "Exercise 2: Training Networks",
      component: lazy(() => import("./exercise/Exercise2")),
    },
  ];
  
  const module = 2;
  
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 2: Neural Network Architectures Exercises"
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

export default ExerciseNeuralNetworks;