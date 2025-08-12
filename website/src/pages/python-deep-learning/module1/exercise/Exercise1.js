import React from "react";
import { Container, Grid, Text, Title } from '@mantine/core';

const Exercise1 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md">
        Exercise 1: Tensor Basics
      </Title>
      <Text mb="lg">
        This exercise covers the fundamentals of creating and manipulating tensors in PyTorch.
      </Text>
      <Grid>
        <Grid.Col span={{ md: 12 }}>
          <Title order={2} mb="md">Instructions</Title>
          <Text>
            Exercise content will be added soon. This is a placeholder for the PyTorch tensor fundamentals exercise.
          </Text>
        </Grid.Col>
      </Grid>
    </Container>
  );
};

export default Exercise1;