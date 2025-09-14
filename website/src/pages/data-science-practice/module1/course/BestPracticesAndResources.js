import React from "react";
import { Container, Grid, Title, Text, List, Code } from '@mantine/core';
import CodeBlock from "components/CodeBlock";

const BestPracticesAndResources = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Best Practices and Resources</Title>
        <Text size="md" mb="md">
          Mastering Git involves understanding and applying best practices that
          can help manage your projects more efficiently. This guide provides
          insights into some critical aspects of using Git effectively.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="good-practices" mb="md">Good Practices in Software Development</Title>
            <Text size="md" mb="md">
              Adopting good software development practices is crucial for the
              success of any project. Here are some key practices:
            </Text>
            <List spacing="sm">
              <List.Item>
                <strong>Define and Follow a Workflow:</strong> Establish and
                adhere to a workflow like Git Flow or feature branching to
                streamline development and collaboration.
              </List.Item>
              <List.Item>
                <strong>Make Small Changes:</strong> Smaller, incremental changes
                are easier to manage and review than large overhauls.
              </List.Item>
              <List.Item>
                <strong>Release Often:</strong> Regular releases help to iterate
                quickly and respond to feedback effectively.
              </List.Item>
              <List.Item>
                <strong>Utilize Pull Requests:</strong> Use pull requests to
                initiate code reviews and merge changes, ensuring quality and
                shared understanding of the codebase.
              </List.Item>
              <List.Item>
                <strong>Conduct Thorough Reviews:</strong> Peer reviews of code
                are essential for maintaining code quality and reducing bugs in
                production.
              </List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="git-tagging" mb="md">Using Git Tags</Title>
            <Text size="md" mb="md">
              Tags in Git are used to create stable releases or to mark a specific
              point in your repository's history as important.
            </Text>
            <List spacing="sm">
              <List.Item>
                <strong>Creating a Tag:</strong> Use <Code>git tag &lt;tagname&gt;</Code> to
                create a lightweight tag, or <Code>git tag -a &lt;tagname&gt; -m
                "message"</Code> for an annotated tag.
              </List.Item>
              <List.Item>
                <strong>Listing Tags:</strong> Use <Code>git tag</Code> to list all tags in
                the repository.
              </List.Item>
              <List.Item>
                <strong>Pushing Tags:</strong> Tags do not automatically transfer
                to the remote repository. Use <Code>git push origin &lt;tagname&gt;</Code> to
                push a single tag or <Code>git push origin --tags</Code> to push all tags.
              </List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <CodeBlock
          code={`git tag v1.0
git tag -a v1.1 -m "Release version 1.1"
git push origin v1.1`}
        />
        <Text size="md" mb="md">
          Tags can help you and your team to refer to specific releases
          without having to remember commit hashes.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="using-git-stash" mb="md">Using Git Stash</Title>
            <Text size="md" mb="md">
              Git stash temporarily shelves (or stashes) changes you've made to
              your working directory, allowing you to work on something else, and
              then come back and re-apply them later on.
            </Text>
            <CodeBlock
              code={`git stash
git stash apply`}
            />
            <Text size="md" mb="md">
              Stashing is handy if you need to quickly switch context and work on
              something else, but you're not ready to commit.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="ci-cd-github-actions" mb="md">CI/CD and GitHub Actions</Title>
            <Text size="md" mb="md">
              Continuous Integration (CI) and Continuous Deployment (CD) are
              practices that automate the integration of code changes and the
              deployment of your application:
            </Text>
            <List spacing="sm">
              <List.Item>
                <strong>CI:</strong> Automatically test and merge code changes.
              </List.Item>
              <List.Item>
                <strong>CD:</strong> Automatically deploy code changes to a
                production or staging environment.
              </List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          GitHub Actions makes CI/CD easy with workflows that can handle
          build, test, and deployment tasks directly within your GitHub
          repository.
        </Text>
        <CodeBlock
          language=""
          code={`name: CI
on: push
jobs: build
runs-on: ubuntu-latest
steps:
- uses: actions/checkout@v2
- name: Run a one-line script
run: python tests.py`}
        />
      </div>
    </Container>
  );
};

export default BestPracticesAndResources;