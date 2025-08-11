import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const ExerciseCloudIntegration = () => {
  const exerciseLinks = [
    // Add links to other exercises as needed
  ];
  const location = useLocation();
  const module = 15;
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 15: Exercise Cloud Integration (with GCP)"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Grid>
            <p>
              In this module, you will practice integrating your applications
              with Google Cloud Platform (GCP).
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
export default ExerciseCloudIntegration;
