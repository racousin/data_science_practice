import React from "react";
import { Container, Grid, Image, Flex, Text, Button, Group, Title, List } from '@mantine/core';
import CodeBlock from "components/CodeBlock";

const GitHubDesktop = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">GitHub Desktop</Title>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="what-is-github-desktop" mb="md">What is GitHub Desktop?</Title>
            <Text size="md" mb="md">
              GitHub Desktop is a free, open-source application that provides a graphical user interface (GUI)
              for Git and GitHub. It simplifies the Git workflow by allowing you to interact with repositories
              through an intuitive visual interface rather than command-line operations.
            </Text>
            <Text size="md" mb="md">
              GitHub Desktop makes it easier for beginners to get started with version control while still
              providing powerful features for experienced developers. It handles complex Git operations behind
              the scenes, allowing you to focus on your code rather than memorizing Git commands.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Flex direction="column" align="center">
          <Image
            src="/assets/data-science-practice/module1/desktop.jpg"
            alt="GitHub Desktop Interface"
            style={{ maxWidth: 'min(700px, 80vw)', height: 'auto' }}
            fluid
          />
          <Text size="sm" c="dimmed">GitHub Desktop Main Interface</Text>
        </Flex>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="installation" mb="md">Installation</Title>
            <Text size="md" mb="md">
              GitHub Desktop is available for Windows and macOS. Choose the appropriate version for your
              operating system:
            </Text>
            <Title order={4} mb="md">Download Links</Title>
            <Group mb="lg">
              <Button
                component="a"
                href="https://desktop.github.com/"
                target="_blank"
                rel="noopener noreferrer"
                variant="filled"
              >
                Download GitHub Desktop
              </Button>
            </Group>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={4} mb="md">System Requirements</Title>
        <List spacing="sm">
          <List.Item>
            <strong>Windows:</strong> Windows 10 64-bit or later
          </List.Item>
          <List.Item>
            <strong>macOS:</strong> macOS 10.13 or later
          </List.Item>
          <List.Item>
            <strong>Linux:</strong> While not officially supported, community-maintained versions are
            available through third-party repositories
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Installation Steps</Title>
        <List type="ordered" spacing="sm">
          <List.Item>Download the installer from the official website</List.Item>
          <List.Item>Run the installer and follow the setup wizard</List.Item>
          <List.Item>Sign in with your GitHub account (or create one if you don't have it)</List.Item>
          <List.Item>Configure your Git settings (name and email)</List.Item>
        </List>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="main-functionalities" mb="md">Main Functionalities</Title>
            <Title order={4} mb="md">1. Repository Management</Title>
            <List spacing="sm">
              <List.Item><strong>Clone repositories:</strong> Download repositories from GitHub to your local machine</List.Item>
              <List.Item><strong>Create repositories:</strong> Initialize new repositories locally or on GitHub</List.Item>
              <List.Item><strong>Add existing repositories:</strong> Import local Git repositories into GitHub Desktop</List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={4} mb="md">2. Making Changes</Title>
        <List spacing="sm">
          <List.Item><strong>Visual diff viewer:</strong> See exactly what changed in your files with side-by-side comparisons</List.Item>
          <List.Item><strong>Selective staging:</strong> Choose specific lines or files to include in a commit</List.Item>
          <List.Item><strong>Commit history:</strong> Browse through the complete history of your repository</List.Item>
          <List.Item><strong>File explorer integration:</strong> Open files directly in your preferred editor</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={4} mb="md">3. Branching and Merging</Title>
        <List spacing="sm">
          <List.Item><strong>Create branches:</strong> Easily create new branches for features or experiments</List.Item>
          <List.Item><strong>Switch branches:</strong> Navigate between branches with a single click</List.Item>
          <List.Item><strong>Merge branches:</strong> Combine changes from different branches</List.Item>
          <List.Item><strong>Resolve conflicts:</strong> Visual tools to help resolve merge conflicts</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={4} mb="md">4. Collaboration Features</Title>
        <List spacing="sm">
          <List.Item><strong>Pull requests:</strong> Create and review pull requests directly from the app</List.Item>
          <List.Item><strong>Sync with remote:</strong> Push and pull changes with a single button</List.Item>
          <List.Item><strong>Fetch updates:</strong> Check for updates from collaborators</List.Item>
          <List.Item><strong>Issue tracking:</strong> Link commits to GitHub issues</List.Item>
        </List>
      </div>
    </Container>
  );
};

export default GitHubDesktop;