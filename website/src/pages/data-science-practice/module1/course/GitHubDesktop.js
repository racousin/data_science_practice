import React from "react";
import { Container, Grid, Image, Flex, Text, Button, Group } from '@mantine/core';
import CodeBlock from "components/CodeBlock";

const GitHubDesktop = () => {
  return (
    <Container fluid>
      <h2>GitHub Desktop</h2>

      {/* What is GitHub Desktop */}
      <Grid>
        <Grid.Col>
          <h3 id="what-is-github-desktop">What is GitHub Desktop?</h3>
          <p>
            GitHub Desktop is a free, open-source application that provides a graphical user interface (GUI)
            for Git and GitHub. It simplifies the Git workflow by allowing you to interact with repositories
            through an intuitive visual interface rather than command-line operations.
          </p>
          <p>
            GitHub Desktop makes it easier for beginners to get started with version control while still
            providing powerful features for experienced developers. It handles complex Git operations behind
            the scenes, allowing you to focus on your code rather than memorizing Git commands.
          </p>

          <Flex direction="column" align="center" mt="md" mb="md">
            <Image
              src="/assets/data-science-practice/module1/desktop.jpg"
              alt="GitHub Desktop Interface"
              style={{ maxWidth: 'min(700px, 80vw)', height: 'auto' }}
              fluid
            />
            <Text size="sm" c="dimmed">GitHub Desktop Main Interface</Text>
          </Flex>
        </Grid.Col>
      </Grid>

      {/* Installation */}
      <Grid>
        <Grid.Col>
          <h3 id="installation">Installation</h3>
          <p>
            GitHub Desktop is available for Windows and macOS. Choose the appropriate version for your
            operating system:
          </p>

          <h4>Download Links</h4>
          <Group mt="md" mb="lg">
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

          <h4>System Requirements</h4>
          <ul>
            <li>
              <strong>Windows:</strong> Windows 10 64-bit or later
            </li>
            <li>
              <strong>macOS:</strong> macOS 10.13 or later
            </li>
            <li>
              <strong>Linux:</strong> While not officially supported, community-maintained versions are
              available through third-party repositories
            </li>
          </ul>

          <h4>Installation Steps</h4>
          <ol>
            <li>Download the installer from the official website</li>
            <li>Run the installer and follow the setup wizard</li>
            <li>Sign in with your GitHub account (or create one if you don't have it)</li>
            <li>Configure your Git settings (name and email)</li>
          </ol>
        </Grid.Col>
      </Grid>

      {/* Main Functionalities */}
      <Grid>
        <Grid.Col>
          <h3 id="main-functionalities">Main Functionalities</h3>

          <h4>1. Repository Management</h4>
          <ul>
            <li><strong>Clone repositories:</strong> Download repositories from GitHub to your local machine</li>
            <li><strong>Create repositories:</strong> Initialize new repositories locally or on GitHub</li>
            <li><strong>Add existing repositories:</strong> Import local Git repositories into GitHub Desktop</li>
          </ul>


          <h4>2. Making Changes</h4>
          <ul>
            <li><strong>Visual diff viewer:</strong> See exactly what changed in your files with side-by-side comparisons</li>
            <li><strong>Selective staging:</strong> Choose specific lines or files to include in a commit</li>
            <li><strong>Commit history:</strong> Browse through the complete history of your repository</li>
            <li><strong>File explorer integration:</strong> Open files directly in your preferred editor</li>
          </ul>

          <h4>3. Branching and Merging</h4>
          <ul>
            <li><strong>Create branches:</strong> Easily create new branches for features or experiments</li>
            <li><strong>Switch branches:</strong> Navigate between branches with a single click</li>
            <li><strong>Merge branches:</strong> Combine changes from different branches</li>
            <li><strong>Resolve conflicts:</strong> Visual tools to help resolve merge conflicts</li>
          </ul>

          <h4>4. Collaboration Features</h4>
          <ul>
            <li><strong>Pull requests:</strong> Create and review pull requests directly from the app</li>
            <li><strong>Sync with remote:</strong> Push and pull changes with a single button</li>
            <li><strong>Fetch updates:</strong> Check for updates from collaborators</li>
            <li><strong>Issue tracking:</strong> Link commits to GitHub issues</li>
          </ul>
        </Grid.Col>
      </Grid>

    </Container>
  );
};

export default GitHubDesktop;