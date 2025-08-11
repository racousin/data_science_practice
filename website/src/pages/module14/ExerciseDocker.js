import React, { lazy } from "react";
import { Row, Col } from 'react-bootstrap';
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
const ExerciseDocker = () => {
  const exerciseLinks = [
    // Add links to other exercises as needed
  ];
  const location = useLocation();
  const module = 14;
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 14: Exercise Docker"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Grid>
            <p>
              In this module, you will practice building, shipping, and running
              applications in containers using Docker.
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
export default ExerciseDocker;
