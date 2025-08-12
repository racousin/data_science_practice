import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const ExerciseRecommendationSystems = () => {
  const exerciseLinks = [
    // Add links to other exercises as needed
  ];const module = 12;
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 12: Exercise Recommendation Systems"
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
export default ExerciseRecommendationSystems;
