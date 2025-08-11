import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const ExerciseTextProcessing = () => {
  const exerciseLinks = [
    {
      to: "/exercise0",
      label: "Exercise 0",
      component: lazy(() => import("pages/module10/exercise/Exercise0")),
    },
    {
      to: "/exercise1",
      label: "Exercise 1",
      component: lazy(() => import("pages/module10/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module10/exercise/Exercise2")),
    },
    {
      to: "/exercise3",
      label: "Exercise 3",
      component: lazy(() => import("pages/module10/exercise/Exercise3")),
    },
  ];
  const location = useLocation();
  const module = 10;
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 10: Exercise Natural Language Processing"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Grid>
            <p>
              In this module, you will practice text processing techniques and
              applications in data science.
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
export default ExerciseTextProcessing;
