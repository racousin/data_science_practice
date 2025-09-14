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

    </>
  );
};

export default GitHubActions;