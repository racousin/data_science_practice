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
  ];
  const location = useLocation();
  const module = 6;
  return (
    <ModuleFrame
      module={6}
      isCourse={false}
      title="Module 6: Exercise Building and Evaluation"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Grid>
            <p>
              In this module, you will practice building and evaluating machine
              learning models.
            </p>
          </Grid>
          <Grid>
            <Grid.Col>
              <p>Last Updated: {"2024-09-20"}</p>
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
export default ExerciseTabularModels;
