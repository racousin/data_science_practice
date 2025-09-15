import React from "react";
import { Container, Title, Text, List, Alert, Space } from '@mantine/core';
import { IconUsersGroup } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";

const Exercise3 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">
        Exercise 3: Collaborating with a Partner
      </Title>

      <Title order={2} mb="md">Overview</Title>
      <Text size="md" mb="lg">
        Work with a classmate to practice real-world Git collaboration. You'll take turns being the repository owner and contributor,
        learning how teams work together on shared code.
      </Text>

      <Alert icon={<IconUsersGroup />} color="blue" mb="lg">
        Pair up with another student. Decide who will be Student A (repository owner) and Student B (collaborator).
        You'll use the same math-docs repository from Exercise 2.
      </Alert>

      <Title order={2} mb="md">Part 1: Setting Up Collaboration</Title>

      <Title order={3} mt="lg" mb="sm">Student A: Add Your Partner</Title>
      <List spacing="sm" mb="md">
        <List.Item>Go to your math-docs repository on GitHub</List.Item>
        <List.Item>Click Settings → Manage access → Add people</List.Item>
        <List.Item>Enter Student B's GitHub username</List.Item>
        <List.Item>Student B will receive an invitation email</List.Item>
      </List>

      <Title order={3} mt="lg" mb="sm">Student B: Accept and Clone</Title>
      <Text size="md" mb="sm">Accept the invitation and clone the repository:</Text>
      <CodeBlock
        code={`git clone https://github.com/studentA-username/math-docs.git
cd math-docs`}
        language="bash"
      />

      <Title order={2} mt="xl" mb="md">Part 2: Collaborative Tasks</Title>

      <Title order={3} mt="lg" mb="sm">Task 1: Student B - Add Division Section</Title>
      <Text size="md" mb="sm">Create a new branch and add division documentation:</Text>
      <CodeBlock
        code={`git checkout -b add-division`}
        language="bash"
      />

      <Text size="md" mb="sm">Add to README.md:</Text>
      <CodeBlock
        code={`### 1.3 Division
Example: 6 / 2 = 3
Note: Division by zero is undefined`}
        language="markdown"
      />

      <Text size="md" mb="sm">Commit and push:</Text>
      <CodeBlock
        code={`git add README.md
git commit -m "Add division section"
git push origin add-division`}
        language="bash"
      />

      <Text size="md" mb="sm">Create a pull request on GitHub for Student A to review.</Text>

      <Title order={3} mt="lg" mb="sm">Task 2: Student A - Review and Merge</Title>
      <List spacing="sm" mb="md">
        <List.Item>Review the pull request on GitHub</List.Item>
        <List.Item>Add a comment: "Looks good! Thanks for adding division."</List.Item>
        <List.Item>Merge the pull request</List.Item>
        <List.Item>Update your local repository:</List.Item>
      </List>

      <CodeBlock
        code={`git checkout main
git pull origin main`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">Task 3: Student A - Add Subtraction Section</Title>
      <Text size="md" mb="sm">Now it's your turn to contribute:</Text>
      <CodeBlock
        code={`git checkout -b add-subtraction`}
        language="bash"
      />

      <Text size="md" mb="sm">Add to README.md:</Text>
      <CodeBlock
        code={`### 1.4 Subtraction
Example: 5 - 2 = 3
The difference between two numbers`}
        language="markdown"
      />

      <Text size="md" mb="sm">Push your changes:</Text>
      <CodeBlock
        code={`git add README.md
git commit -m "Add subtraction section"
git push origin add-subtraction`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">Task 4: Student B - Review and Update</Title>
      <List spacing="sm" mb="md">
        <List.Item>Review Student A's pull request</List.Item>
        <List.Item>Approve and comment on the changes</List.Item>
        <List.Item>After merge, update your local copy:</List.Item>
      </List>

      <CodeBlock
        code={`git checkout main
git pull origin main`}
        language="bash"
      />

      <Title order={2} mt="xl" mb="md">Part 3: Simultaneous Work</Title>

      <Text size="md" mb="md">
        Both students work at the same time on different sections to practice handling parallel development.
      </Text>

      <Title order={3} mt="lg" mb="sm">Student A: Create examples.md</Title>
      <CodeBlock
        code={`git checkout -b add-examples
echo "# Additional Examples" > examples.md
git add examples.md
git commit -m "Add examples file"
git push origin add-examples`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">Student B: Create formulas.md</Title>
      <CodeBlock
        code={`git checkout -b add-formulas
echo "# Mathematical Formulas" > formulas.md
git add formulas.md
git commit -m "Add formulas file"
git push origin add-formulas`}
        language="bash"
      />

      <Text size="md" mb="md">
        Both students create pull requests. Student A merges both in sequence, demonstrating how Git handles parallel work.
      </Text>

      <Title order={2} mt="xl" mb="md">Verification</Title>
      <Text size="md" mb="sm">After completing all tasks, your repository should contain:</Text>
      <List spacing="xs">
        <List.Item>README.md with all four arithmetic operations</List.Item>
        <List.Item>examples.md file</List.Item>
        <List.Item>formulas.md file</List.Item>
        <List.Item>Multiple merged pull requests in the history</List.Item>
      </List>

      <Space h="xl" />
    </Container>
  );
};

export default Exercise3;