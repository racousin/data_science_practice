import React from "react";
import { Container, Grid, Text, Title } from '@mantine/core';

const Exercise1 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md">
        Exercise 1: Advanced Architectures
      </Title>
      <Text mb="lg">
        This exercise covers advanced deep learning architectures and techniques.
      </Text>
      <Grid>
        <Grid.Col span={{ md: 12 }}>
          <Title order={2} mb="md">Instructions</Title>
          <Text>
            Exercise content will be added soon. This is a placeholder for the advanced architectures exercise.
          </Text>
        </Grid.Col>
      </Grid>
    </Container>
  );
};

export default Exercise1;