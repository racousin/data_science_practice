import React, { lazy } from "react";
import { Row, Col } from 'react-bootstrap';
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const ExerciseDataPreprocessing = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module5/exercise/Exercise1")),
    },
  ];
  const location = useLocation();
  const module = 5;
  return (
    <ModuleFrame
      module={5}
      isCourse={false}
      title="Module 5: Data Preprocessing"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
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
export default ExerciseDataPreprocessing;
