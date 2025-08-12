import React from "react";
import { Container, Grid, Text, Title } from '@mantine/core';

const Exercise1 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md">
        Exercise 1: Model Deployment
      </Title>
      <Text mb="lg">
        This exercise covers deploying deep learning models to production environments.
      </Text>
      <Grid>
        <Grid.Col span={{ md: 12 }}>
          <Title order={2} mb="md">Instructions</Title>
          <Text>
            Exercise content will be added soon. This is a placeholder for the model deployment exercise.
          </Text>
        </Grid.Col>
      </Grid>
    </Container>
  );
};

export default Exercise1;