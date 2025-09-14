import React from "react";
import { Container, Title, Text, List, Alert, Space, Badge, Group } from '@mantine/core';
import { IconRocket, IconBulb, IconGitPullRequest } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";

const Exercise5 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">
        Exercise 5: Your First Open Source Contribution 🚀
      </Title>

      <Alert icon={<IconRocket />} color="violet" mb="lg">
        Welcome to the real world! It's time to contribute to one of the most influential
        machine learning libraries in the world: scikit-learn.
      </Alert>

      <Title order={2} mb="md">Why This Matters</Title>
      <Text size="md" mb="lg">
        Contributing to open source is how the software world evolves. Your contribution,
        no matter how small, becomes part of tools used by millions of data scientists worldwide.
        Today, you start your journey as an open source contributor.
      </Text>

      <Title order={2} mb="md">Part 1: Understanding the Project</Title>

      <Title order={3} mt="lg" mb="sm">Step 1: Read the Contributing Guide</Title>
      <Text size="md" mb="sm">
        Visit and carefully read: <a href="https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">
        scikit-learn Contributing Guide</a>
      </Text>
      <Text size="md" mb="md">
        Pay special attention to:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>Code of Conduct</List.Item>
        <List.Item>How to report issues</List.Item>
        <List.Item>Coding guidelines</List.Item>
        <List.Item>Documentation standards</List.Item>
      </List>

      <Title order={3} mt="lg" mb="sm">Step 2: Explore Good First Issues</Title>
      <Text size="md" mb="sm">
        Browse: <a href="https://github.com/scikit-learn/scikit-learn/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22" target="_blank" rel="noopener noreferrer">
        Good First Issues</a>
      </Text>
      <Text size="md" mb="md">
        Look for issues with these labels:
      </Text>
      <Group mb="md">
        <Badge color="green">good first issue</Badge>
        <Badge color="blue">Documentation</Badge>
        <Badge color="orange">Easy</Badge>
      </Group>

      <Title order={2} mt="xl" mb="md">Part 2: Setting Up Your Environment</Title>

      <Title order={3} mt="lg" mb="sm">Step 3: Fork and Clone</Title>
      <Text size="md" mb="sm">Fork scikit-learn to your GitHub account, then:</Text>
      <CodeBlock
        code={`git clone https://github.com/YOUR-USERNAME/scikit-learn.git
cd scikit-learn
git remote add upstream https://github.com/scikit-learn/scikit-learn.git`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">Step 4: Create a Development Environment</Title>
      <CodeBlock
        code={`# Create a virtual environment
python -m venv sklearn-dev
source sklearn-dev/bin/activate  # On Windows: sklearn-dev\\Scripts\\activate

# Install in development mode
pip install -e .`}
        language="bash"
      />

      <Title order={2} mt="xl" mb="md">Part 3: Finding Your Contribution</Title>

      <Alert icon={<IconBulb />} color="yellow" mb="lg">
        Start small! Your first contribution doesn't need to be code. Consider:
        Documentation improvements, fixing typos, adding examples, or improving error messages.
      </Alert>

      <Title order={3} mt="lg" mb="sm">Suggested First Contributions</Title>

      <Title order={4} mt="md" mb="sm">Option A: Documentation Improvement</Title>
      <List spacing="sm" mb="md">
        <List.Item>Find a function with unclear documentation</List.Item>
        <List.Item>Improve parameter descriptions or add examples</List.Item>
        <List.Item>Fix grammatical errors or clarify explanations</List.Item>
      </List>

      <Title order={4} mt="md" mb="sm">Option B: Add a Code Example</Title>
      <List spacing="sm" mb="md">
        <List.Item>Find a function lacking usage examples</List.Item>
        <List.Item>Write a clear, practical example</List.Item>
        <List.Item>Ensure it follows the project's style guide</List.Item>
      </List>

      <Title order={4} mt="md" mb="sm">Option C: Fix a Simple Bug</Title>
      <List spacing="sm" mb="md">
        <List.Item>Choose an issue labeled "good first issue"</List.Item>
        <List.Item>Comment on the issue: "I'd like to work on this"</List.Item>
        <List.Item>Wait for maintainer approval before starting</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md">Part 4: Making Your Contribution</Title>

      <Title order={3} mt="lg" mb="sm">Step 5: Create a Feature Branch</Title>
      <CodeBlock
        code={`git checkout main
git pull upstream main
git checkout -b descriptive-branch-name`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">Step 6: Make Your Changes</Title>
      <Text size="md" mb="sm">Follow these principles:</Text>
      <List spacing="sm" mb="md">
        <List.Item>Make minimal, focused changes</List.Item>
        <List.Item>Follow the existing code style</List.Item>
        <List.Item>Write clear commit messages</List.Item>
        <List.Item>Test your changes locally</List.Item>
      </List>

      <Title order={3} mt="lg" mb="sm">Step 7: Run Tests (if applicable)</Title>
      <CodeBlock
        code={`# Run tests for your changes
pytest sklearn/tests/test_your_module.py

# Check code style
flake8 sklearn/your_file.py`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">Step 8: Submit Your Pull Request</Title>
      <CodeBlock
        code={`git add .
git commit -m "DOC: Improve documentation for function_name"
git push origin descriptive-branch-name`}
        language="bash"
      />

      <Text size="md" mb="md">
        Go to GitHub and create a pull request. In your PR description:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>Reference the issue number (if applicable)</List.Item>
        <List.Item>Describe what you changed and why</List.Item>
        <List.Item>Show before/after if it's a visual change</List.Item>
        <List.Item>Be patient and responsive to feedback</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md">Part 5: The Review Process</Title>

      <Title order={3} mt="lg" mb="sm">What to Expect</Title>
      <List spacing="sm" mb="md">
        <List.Item>Maintainers will review your PR (this may take days or weeks)</List.Item>
        <List.Item>They may request changes - this is normal and helpful!</List.Item>
        <List.Item>Address feedback professionally and promptly</List.Item>
        <List.Item>Once approved, your code becomes part of scikit-learn!</List.Item>
      </List>

      <Alert icon={<IconGitPullRequest />} color="green" mb="lg">
        Pro tip: While waiting for review, help by reviewing other PRs.
        This builds your reputation and understanding of the codebase.
      </Alert>

      <Title order={2} mt="xl" mb="md">Celebration Milestone 🎉</Title>
      <Text size="md" mb="md">
        When your PR is merged:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>You become an official scikit-learn contributor</List.Item>
        <List.Item>Your name appears in the contributor list</List.Item>
        <List.Item>Add "Open Source Contributor to scikit-learn" to your resume</List.Item>
        <List.Item>Share your achievement on LinkedIn/Twitter</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md">Alternative Projects for First Contributors</Title>
      <Text size="md" mb="md">
        If scikit-learn feels overwhelming, consider these beginner-friendly alternatives:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item><strong>NumPy</strong>: Focus on documentation improvements</List.Item>
        <List.Item><strong>Pandas</strong>: Many "good first issue" opportunities</List.Item>
        <List.Item><strong>Matplotlib</strong>: Always needs example improvements</List.Item>
        <List.Item><strong>First Contributions</strong>: A practice repository for beginners</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md">Your Journey Starts Here</Title>
      <Text size="md" mb="lg">
        Remember: Every expert was once a beginner. The maintainers were once making their
        first contribution too. Be brave, be respectful, and welcome to the open source community!
      </Text>

      <Alert color="violet" mb="lg">
        <strong>Assignment:</strong> Submit the URL of your pull request (even if still under review)
        as proof of your first open source contribution attempt. Whether it gets merged or not,
        you've taken the crucial first step!
      </Alert>

      <Space h="xl" />
    </Container>
  );
};

export default Exercise5;