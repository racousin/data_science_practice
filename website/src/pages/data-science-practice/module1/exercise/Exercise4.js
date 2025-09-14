import React from "react";
import { Container, Title, Text, List, Alert, Space } from '@mantine/core';
import { IconRobot } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";

const Exercise4 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">
        Exercise 4: Automating Tasks with GitHub Actions
      </Title>

      <Title order={2} mb="md">Introduction</Title>
      <Text size="md" mb="lg">
        GitHub Actions automates repetitive tasks in your repository. You'll create workflows that run automatically
        when specific events occur, like pushing code or creating pull requests.
      </Text>

      <Alert icon={<IconRobot />} color="green" mb="lg">
        Continue using your math-docs repository. If you worked on a partner's repo in Exercise 3,
        return to your own math-docs repository for this exercise.
      </Alert>

      <Title order={2} mb="md">Part 1: Documentation Checker</Title>

      <Title order={3} mt="lg" mb="sm">Task 1: Create Your First Workflow</Title>
      <Text size="md" mb="sm">
        Create a workflow that checks if your documentation files exist and are not empty.
      </Text>

      <Text size="md" mb="sm">Create the workflow directory structure:</Text>
      <CodeBlock
        code={`mkdir -p .github/workflows`}
        language="bash"
      />

      <Text size="md" mb="sm">Create .github/workflows/check-docs.yml:</Text>
      <CodeBlock
        code={`name: Documentation Check

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  check-files:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Check README exists
      run: |
        if [ ! -f README.md ]; then
          echo "❌ README.md is missing!"
          exit 1
        fi
        echo "✅ README.md exists"

    - name: Check README not empty
      run: |
        if [ ! -s README.md ]; then
          echo "❌ README.md is empty!"
          exit 1
        fi
        echo "✅ README.md has content"
        echo "File size: $(wc -c < README.md) bytes"`}
        language="yaml"
      />

      <Text size="md" mb="sm">Commit and push to trigger the action:</Text>
      <CodeBlock
        code={`git add .github/workflows/check-docs.yml
git commit -m "Add documentation check workflow"
git push origin main`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">Task 2: View Your Action Running</Title>
      <List spacing="sm" mb="md">
        <List.Item>Go to your repository on GitHub</List.Item>
        <List.Item>Click the "Actions" tab</List.Item>
        <List.Item>Watch your workflow run automatically</List.Item>
        <List.Item>Click on the workflow run to see detailed logs</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md">Part 2: Markdown Formatter</Title>

      <Title order={3} mt="lg" mb="sm">Task 3: Create a Formatting Workflow</Title>
      <Text size="md" mb="sm">
        Add a workflow that formats and validates your Markdown files.
      </Text>

      <Text size="md" mb="sm">Create .github/workflows/format-markdown.yml:</Text>
      <CodeBlock
        code={`name: Format Markdown

on:
  push:
    paths:
      - '**.md'

jobs:
  format:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Count Markdown files
      run: |
        count=$(find . -name "*.md" -type f | wc -l)
        echo "📝 Found $count Markdown file(s)"

    - name: Check Markdown structure
      run: |
        echo "Checking Markdown files for basic structure..."
        for file in $(find . -name "*.md" -type f); do
          echo "Checking $file"
          if grep -q "^# " "$file"; then
            echo "✅ $file has a main heading"
          else
            echo "⚠️  $file missing main heading"
          fi
        done

    - name: Generate table of contents
      run: |
        echo "## Repository Structure" > TOC.md
        echo "" >> TOC.md
        echo "### Markdown Files:" >> TOC.md
        find . -name "*.md" -type f | while read file; do
          echo "- $file" >> TOC.md
        done
        cat TOC.md`}
        language="yaml"
      />

      <Title order={2} mt="xl" mb="md">Part 3: Welcome Bot</Title>

      <Title order={3} mt="lg" mb="sm">Task 4: Create an Issue Welcome Message</Title>
      <Text size="md" mb="sm">
        Create a workflow that automatically responds when someone opens an issue.
      </Text>

      <Text size="md" mb="sm">Create .github/workflows/welcome.yml:</Text>
      <CodeBlock
        code={`name: Welcome New Contributors

on:
  issues:
    types: [opened]

jobs:
  welcome:
    runs-on: ubuntu-latest

    steps:
    - name: Post welcome message
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: '👋 Thanks for opening this issue! We appreciate your contribution to the math-docs project.'
          })`}
        language="yaml"
      />

      <Title order={3} mt="lg" mb="sm">Task 5: Test Your Welcome Bot</Title>
      <List spacing="sm" mb="md">
        <List.Item>Push the welcome.yml file to your repository</List.Item>
        <List.Item>Go to the "Issues" tab on GitHub</List.Item>
        <List.Item>Create a new issue with any title</List.Item>
        <List.Item>Watch as the bot automatically responds!</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md">Part 4: Scheduled Documentation Update</Title>

      <Title order={3} mt="lg" mb="sm">Task 6: Create a Scheduled Workflow</Title>
      <Text size="md" mb="sm">
        Create a workflow that runs on a schedule to update documentation timestamps.
      </Text>

      <Text size="md" mb="sm">Create .github/workflows/scheduled-update.yml:</Text>
      <CodeBlock
        code={`name: Weekly Documentation Review

on:
  schedule:
    # Runs every Monday at 9 AM UTC
    - cron: '0 9 * * 1'
  workflow_dispatch:  # Allows manual trigger

jobs:
  update-timestamp:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Update last review date
      run: |
        echo "## Last Review" > LAST_REVIEW.md
        echo "Documentation last reviewed: $(date)" >> LAST_REVIEW.md
        cat LAST_REVIEW.md

    - name: Check documentation age
      run: |
        echo "Checking when files were last modified..."
        find . -name "*.md" -type f -exec sh -c '
          echo "File: $1"
          echo "Last modified: $(stat -c %y "$1" 2>/dev/null || stat -f %Sm "$1")"
        ' sh {} \\;`}
        language="yaml"
      />

      <Alert color="blue" mb="lg">
        Note: The scheduled workflow won't run immediately. Use "Actions" → "Weekly Documentation Review" → "Run workflow"
        to test it manually.
      </Alert>

      <Title order={2} mt="xl" mb="md">Part 5: Status Badge</Title>

      <Title order={3} mt="lg" mb="sm">Task 7: Add a Status Badge</Title>
      <Text size="md" mb="sm">
        Display your workflow status in your README. Add this line to the top of your README.md:
      </Text>

      <CodeBlock
        code={`![Documentation Check](https://github.com/YOUR-USERNAME/math-docs/actions/workflows/check-docs.yml/badge.svg)`}
        language="markdown"
      />

      <Text size="md" mb="sm">
        Replace YOUR-USERNAME with your GitHub username. This badge will show if your documentation checks are passing.
      </Text>

      <Title order={2} mt="xl" mb="md">Verification</Title>
      <Text size="md" mb="sm">After completing all tasks, verify:</Text>
      <List spacing="xs">
        <List.Item>Go to the "Actions" tab - you should see 4 workflows</List.Item>
        <List.Item>Each workflow should have at least one successful run</List.Item>
        <List.Item>Your README should display a status badge</List.Item>
        <List.Item>Creating an issue triggers the welcome message</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md">Understanding the Benefits</Title>
      <Text size="md" mb="md">
        You've now automated several tasks that would normally require manual work:
      </Text>
      <List spacing="xs">
        <List.Item>Automatic validation of documentation</List.Item>
        <List.Item>Consistent formatting checks</List.Item>
        <List.Item>Automated responses to contributors</List.Item>
        <List.Item>Scheduled maintenance tasks</List.Item>
      </List>

      <Space h="xl" />
    </Container>
  );
};

export default Exercise4;