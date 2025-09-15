import React from "react";
import { Container, Grid, Image, Title, Text, List } from '@mantine/core';
const Introduction = () => {
  return (
    <Container fluid>
      <Title order={2} mb="md">Introduction</Title>
      <Grid>
        <Grid.Col span={{ md: 12 }}>
          <Title order={4} mb="sm">Why Use Python Environments?</Title>
          <Text size="md" mb="md">
            Python environments are isolated contexts where Python packages and
            dependencies are installed. This isolation prevents version
            conflicts and ensures that projects can be developed and tested in
            settings that closely mimic their production environments.
          </Text>
          <Title order={4} mb="sm">The Importance of Package Management</Title>
          <Text size="md" mb="md">
            Package management involves organizing, installing, and maintaining
            software libraries that projects depend on. Python's package
            ecosystem includes thousands of third-party modules available on the
            Python Package Index (PyPI), which can be managed using tools like
            pip and conda.
          </Text>
          <Title order={4} mb="sm">Key Tools for Python Environments and Package Management</Title>
          <Text size="md" mb="md">
            The following tools are commonly used for managing Python
            environments and packages:
          </Text>
          <List spacing="sm">
            <List.Item>
              <Text component="span" fw={700}>pip:</Text> Python's standard package-management system
              used to install and manage software packages.
            </List.Item>
            <List.Item>
              <Text component="span" fw={700}>virtualenv:</Text> A tool to create isolated Python
              environments.
            </List.Item>
            <List.Item>
              <Text component="span" fw={700}>conda:</Text> An open-source package management system
              and environment management system.
            </List.Item>
            <List.Item>
              <Text component="span" fw={700}>Poetry:</Text> An open-source package management system
              and environment management system.
            </List.Item>
            <List.Item>
              <Text component="span" fw={700}>Pipenv:</Text> An open-source package management system
              and environment management system.
            </List.Item>
          </List>
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col span={{ md: 10 }}>
          <Image
            src="/assets/data-science-practice/module2/python.png"
            alt="Q-values"
            fluid
          />
        </Grid.Col>
      </Grid>
    </Container>
  );
};
export default Introduction;
