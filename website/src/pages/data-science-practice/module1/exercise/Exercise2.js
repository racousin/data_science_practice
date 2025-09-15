import React from "react";
import { Container, Title, Text, List, Space } from '@mantine/core';
import CodeBlock from "components/CodeBlock";

const Exercise2 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">
        Exercise 2: Managing Branches and Correcting Errors
      </Title>

      <Title order={2} mb="md">Scenario</Title>
      <Text size="md" mb="lg">
        You're documenting a math library when you discover an error in your initial commit.
        Learn to fix mistakes while preserving ongoing work using Git branches.
      </Text>

      <Title order={2} mb="md">Tasks</Title>

      <Title order={3} mt="lg" mb="sm">1. Repository Setup</Title>
      <Text size="md" mb="sm">Create and clone your repository:</Text>
      <CodeBlock
        code={`git clone git@github.com:your-username/math-docs.git
cd math-docs`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">2. Initial Documentation</Title>
      <Text size="md" mb="sm">Create README.md with this content (note the deliberate error):</Text>
      <CodeBlock
        code={`# Mathematics Library Documentation

## 1. Basic Arithmetic
### 1.1 Addition
Example: 2 + 3 = 5

### 1.2 Multiplication
Example: 2 * 3 = 5  # Error: should be 6`}
        language="markdown"
      />

      <Text size="md" mb="sm">Commit and push:</Text>
      <CodeBlock
        code={`git add README.md
git commit -m "Add initial documentation"
git push origin main`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">3. New Feature Branch</Title>
      <Text size="md" mb="sm">Start working on advanced operations:</Text>
      <CodeBlock
        code={`git checkout -b advanced-operations`}
        language="bash"
      />

      <Text size="md" mb="sm">Add to README.md:</Text>
      <CodeBlock
        code={`## 2. Advanced Operations
### 2.1 Exponentiation
Example: 2^3 = 8`}
        language="markdown"
      />

      <Text size="md" mb="sm">Commit your work:</Text>
      <CodeBlock
        code={`git add README.md
git commit -m "Add advanced operations"`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">4. Fix the Error</Title>
      <Text size="md" mb="sm">Create a bugfix branch from main:</Text>
      <CodeBlock
        code={`git checkout main
git checkout -b fix-multiplication`}
        language="bash"
      />

      <Text size="md" mb="sm">Correct the multiplication example to "2 * 3 = 6" and push:</Text>
      <CodeBlock
        code={`git add README.md
git commit -m "Fix multiplication error"
git push origin fix-multiplication`}
        language="bash"
      />

      <Text size="md" mb="sm">Create and merge a pull request on GitHub.</Text>

      <Title order={3} mt="lg" mb="sm">5. Update Feature Branch</Title>
      <Text size="md" mb="sm">Incorporate the fix into your feature branch:</Text>
      <CodeBlock
        code={`git checkout advanced-operations
git merge main
git push origin advanced-operations`}
        language="bash"
      />

      <Title order={2} mt="xl" mb="md">Expected Result</Title>
      <Text size="md" mb="sm">Your advanced-operations branch should now have:</Text>
      <List spacing="xs">
        <List.Item>Corrected multiplication: 2 * 3 = 6</List.Item>
        <List.Item>New advanced operations section</List.Item>
        <List.Item>Clean merge history</List.Item>
      </List>

      <Space h="xl" />
    </Container>
  );
};

export default Exercise2;