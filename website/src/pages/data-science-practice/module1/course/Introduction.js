import React from "react";
import { Container, Grid, Image, Title, Text, List, Anchor } from '@mantine/core';
const Introduction = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Introduction</Title>
        <Image
          src="/assets/data-science-practice/module1/merge.gif"
          alt="Git_Fetch_Merge_Pull"
          fluid
          style={{ maxWidth: 'min(600px, 60vw)', height: 'auto' }}
        />
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <Text size="md" mb="md">
              Git is a distributed version control system widely used to
              coordinate work among programmers. It tracks changes in source code
              during software development, allowing for efficient collaboration
              and historical referencing.
            </Text>
            <Text size="md" mb="md">
              Developed by Linus Torvalds in 2005 for Linux kernel development,
              Git has since become essential for managing projects ranging from
              small teams to large enterprises. It supports non-linear development
              through its robust branching and merging capabilities, enabling
              multiple parallel workflows.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Why Use Version Control?</Title>
        <Text size="md" mb="md">
          Version control systems are fundamental in software development for
          maintaining a clear history of code changes, facilitating
          collaborative adjustments, and ensuring that earlier versions of
          work can be retrieved. This is crucial in complex projects where
          tracking the contributions of each team member is necessary for
          effective progression.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Git Platforms in Modern Development</Title>
        <Text size="md" mb="md">
          Git's impact extends beyond just tracking changes. It integrates
          with various services to enhance project management capabilities.
          Here are a few key platforms:
        </Text>
        <List spacing="sm">
          <List.Item>
            <Anchor href="https://github.com" target="_blank">GitHub</Anchor>
          </List.Item>
          <List.Item>
            <Anchor href="https://gitlab.com" target="_blank">GitLab</Anchor>
          </List.Item>
          <List.Item>
            <Anchor href="https://bitbucket.org" target="_blank">Bitbucket</Anchor>
          </List.Item>
          <List.Item>
            <Anchor href="https://aws.amazon.com/codecommit/" target="_blank">AWS CodeCommit</Anchor>
          </List.Item>
          <List.Item>
            <Anchor href="https://cloud.google.com/source-repositories" target="_blank">
              Google Cloud Source Repositories
            </Anchor>
          </List.Item>
          <List.Item>
            <Anchor href="https://azure.microsoft.com/en-us/services/devops/repos/" target="_blank">
              Azure Repos
            </Anchor>
          </List.Item>
        </List>
      </div>
    </Container>
  );
};
export default Introduction;
