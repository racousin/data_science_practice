import React from "react";
import { Container, Grid, Title, Text, List, Alert } from '@mantine/core';
import { IconAlertCircle } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";
import EvaluationModal from "components/EvaluationModal";
const Exercise1 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md">Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span>: Creating a Python Package</Title>

      <Text size="md" mb="md">
        In this exercise, you will create a Python package named `mysupertools`
        with a module that contains a function to multiply two values. This
        function will return the product if both arguments are numbers, and the
        string "error" otherwise.
      </Text>
      <Title order={2} mb="md" mt="xl">Instructions</Title>

      <Text size="md" mb="md">
        Follow these steps to create your Python package with the correct structure:
      </Text>

      <Title order={3} mb="sm">1. Directory Structure</Title>
      <Text size="md" mb="md">
        Your final directory structure should look like this:
      </Text>
      <CodeBlock
        code={`$username/module2/mysupertools/
                ├── pyproject.toml
                └── mysupertools/
                    ├── __init__.py
                    └── tool/
                        ├── __init__.py
                        └── operation_a_b.py`}
      />

      <Title order={3} mb="sm" mt="lg">2. Implementation Function</Title>
      <Text size="md" mb="md">
        The <code>operation_a_b.py</code> file should contain a function{" "}
        <code>multiply(a, b)</code> that returns <i>a × b</i> if both arguments are numbers,
        otherwise returns the string "error".
      </Text>

      <CodeBlock
        code={`def multiply(a, b):
    """
    Multiply two values if they are both numbers.

    Args:
        a: First value
        b: Second value

    Returns:
        Product of a and b if both are numbers, "error" otherwise
    """
    # Your implementation here
    pass`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">3. Package Configuration</Title>
      <Text size="md" mb="md">
        Create a <code>pyproject.toml</code> file in the <code>mysupertools</code> directory:
      </Text>
      <CodeBlock
        code={`[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mysupertools"
version = "0.1.0"
description = "A simple tool package with multiplication function"
authors = [{name = "Your Name", email = "your.email@example.com"}]
requires-python = ">=3.8"
classifiers = [
]

[tool.setuptools.packages.find]
where = ["."]
include = ["mysupertools*"]`}
        language="toml"
      />

      <Title order={3} mb="sm" mt="lg">4. Package Initialization</Title>
      <Text size="md" mb="md">
        Don't forget the <code>__init__.py</code> files inside both the{" "}
        <code>mysupertools</code> and <code>tool</code> directories to make them Python packages.
        These files can be empty or contain package initialization code.
      </Text>
      <Title order={2} mb="md" mt="xl">Testing Your Code</Title>

      <Text size="md" mb="md">
        To ensure your package is working correctly, follow these steps to test your implementation:
      </Text>

      <Title order={3} mb="sm">Installation</Title>
      <Text size="md" mb="md">
        First, install your package in development mode. It's recommended to do this in a virtual environment:
      </Text>
      <CodeBlock
        code={`# Navigate to your package directory
cd $username/module2/mysupertools

# Install in development mode
pip install -e .`}
        language="bash"
      />

      <Title order={3} mb="sm" mt="lg">Testing the Function</Title>
      <Text size="md" mb="md">
        Open a Python session and test your function:
      </Text>
      <CodeBlock
        code={`python`}
        language="bash"
      />
      <CodeBlock
        code={`from mysupertools.tool.operation_a_b import multiply

# Test with numbers
assert multiply(4, 5) == 20
assert multiply(2.5, 4) == 10.0
assert multiply(-3, 7) == -21

# Test with non-numbers
assert multiply("a", 5) == "error"
assert multiply(4, "b") == "error"
assert multiply("hello", "world") == "error"

print("All tests passed!")`}
        language="python"
      />

      <Alert icon={<IconAlertCircle />} color="yellow" mt="md">
        Note: The function should return exactly "error" as a string for invalid inputs,
        not perform string multiplication like "aaaaa".
      </Alert>

      <Title order={2} mb="md" mt="xl">Submission Guidelines</Title>

      <Text size="md" mb="md">
        When submitting your work, follow these important guidelines:
      </Text>

      <List spacing="sm" size="md" mb="md">
        <List.Item>Only push the package source code, not the build artifacts</List.Item>
        <List.Item>Your submission should include the <code>mysupertools/</code> directory with source files</List.Item>
        <List.Item>Do NOT include the <code>build/</code> directory if it exists</List.Item>
        <List.Item>Do NOT include the <code>mysupertools.egg-info/</code> directory if it exists</List.Item>
      </List>

      <Alert icon={<IconAlertCircle />} color="red" mb="md">
        <strong>Important:</strong> Only push the package source code (the <code>mysupertools/</code> directory),
        not the build files (<code>build</code> and <code>mysupertools.egg-info</code> directories).
        These are generated during installation and should not be committed to the repository.
      </Alert>

      <Text size="md" mb="md">
        Your final submission should have this structure in your repository:
      </Text>

      <CodeBlock
        code={`$username/module2/mysupertools/
├── pyproject.toml
└── mysupertools/
    ├── __init__.py
    └── tool/
        ├── __init__.py
        └── operation_a_b.py`}
      />

      <Grid>
        <Grid.Col>
          <EvaluationModal module={2} />
        </Grid.Col>
      </Grid>
    </Container>
  );
};
export default Exercise1;
