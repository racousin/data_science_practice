import React from 'react';
import { Title, Text, List, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const GitHubActions = () => {
  return (
    <>
      <div data-slide>
        <Title order={1} mb="lg">GitHub Actions</Title>
        <Text size="lg" mb="md">
          GitHub Actions is a continuous integration and continuous delivery (CI/CD) platform that allows you to automate your build, test, and deployment pipeline.
        </Text>
        <Text size="md">
          You can create workflows that run when specific events occur in your repository, such as pushing code or creating a pull request.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Key Concepts</Title>
        <List spacing="md">
          <List.Item>
            <strong>Workflows</strong> - Automated processes defined in YAML files
          </List.Item>
          <List.Item>
            <strong>Events</strong> - Triggers that start workflows (push, pull request, schedule)
          </List.Item>
          <List.Item>
            <strong>Jobs</strong> - Sets of steps that execute on the same runner
          </List.Item>
          <List.Item>
            <strong>Steps</strong> - Individual tasks that run commands or actions
          </List.Item>
          <List.Item>
            <strong>Runners</strong> - Servers that execute your workflows
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Basic Workflow Structure</Title>
        <Text mb="md">
          Workflows are defined in <Code>.github/workflows/</Code> directory using YAML format:
        </Text>
        <CodeBlock
          code={`name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest`}
          language="yaml"
        />
        <Text size="sm" mt="md">
          This workflow runs on every push and pull request, using an Ubuntu runner.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Common Workflow Triggers</Title>
        <List spacing="md">
          <List.Item>
            <Code>push</Code> - Triggered when code is pushed to the repository
          </List.Item>
          <List.Item>
            <Code>pull_request</Code> - Triggered when a pull request is opened or updated
          </List.Item>
          <List.Item>
            <Code>schedule</Code> - Run workflows on a schedule using cron syntax
          </List.Item>
          <List.Item>
            <Code>workflow_dispatch</Code> - Manually trigger workflows from GitHub UI
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Example: Python Testing Workflow</Title>
        <Text mb="md">
          A complete workflow that tests Python code:
        </Text>
        <CodeBlock
          code={`name: Python Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest`}
          language="yaml"
        />
        <CodeBlock
          code={`    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'`}
          language="yaml"
        />
        <CodeBlock
          code={`    - run: pip install -r requirements.txt
    - run: pytest tests/`}
          language="yaml"
        />
      </div>

      <div data-slide>
        <Title order={2} mb="md">Using Actions from the Marketplace</Title>
        <Text mb="md">
          GitHub provides pre-built actions that you can use in your workflows:
        </Text>
        <List spacing="md">
          <List.Item>
            <Code>actions/checkout</Code> - Check out your repository code
          </List.Item>
          <List.Item>
            <Code>actions/setup-python</Code> - Set up Python environment
          </List.Item>
          <List.Item>
            <Code>actions/upload-artifact</Code> - Upload build artifacts
          </List.Item>
          <List.Item>
            <Code>actions/cache</Code> - Cache dependencies for faster builds
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Environment Variables and Secrets</Title>
        <Text mb="md">
          Store sensitive data securely and access it in workflows:
        </Text>
        <CodeBlock
          code={`env:
  MY_VAR: "public value"
  API_KEY: \${{ secrets.API_KEY }}`}
          language="yaml"
        />
        <Text size="sm" mt="md">
          Secrets are encrypted and never exposed in logs. Define them in repository settings.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Matrix Builds</Title>
        <Text mb="md">
          Test across multiple versions or configurations:
        </Text>
        <CodeBlock
          code={`strategy:
  matrix:
    python: ['3.8', '3.9', '3.10']
    os: [ubuntu-latest, windows-latest]`}
          language="yaml"
        />
        <Text size="sm" mt="md">
          This creates 6 jobs testing all combinations of Python versions and operating systems.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Conditional Execution</Title>
        <Text mb="md">
          Control when steps or jobs run using conditions:
        </Text>
        <CodeBlock
          code={`- name: Deploy
  if: github.ref == 'refs/heads/main'
  run: ./deploy.sh`}
          language="yaml"
        />
        <Text size="sm" mt="md">
          This step only runs when pushing to the main branch.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Best Practices</Title>
        <List spacing="md">
          <List.Item>
            Keep workflows simple and focused on a single purpose
          </List.Item>
          <List.Item>
            Use caching to speed up workflows (dependencies, build artifacts)
          </List.Item>
          <List.Item>
            Set timeouts to prevent workflows from running indefinitely
          </List.Item>
          <List.Item>
            Use secrets for sensitive data, never hardcode credentials
          </List.Item>
          <List.Item>
            Test workflows in a separate branch before merging to main
          </List.Item>
        </List>
      </div>
    </>
  );
};

export default GitHubActions;