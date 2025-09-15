import React from "react";
import { Container, Grid, Title, Text, List, Alert } from '@mantine/core';
import { IconAlertCircle } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";
import EvaluationModal from "components/EvaluationModal";

const Exercise2 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md">Exercise 2<span style={{color: 'red', fontWeight: 'bold'}}>*</span>: Understanding Unit Tests and CI</Title>

      <Text size="md" mb="md">
        Learn why unit tests and continuous integration are essential for software development
        by building a simple <code>mathtools</code> package with automated testing.
      </Text>

      <Title order={2} mb="md" mt="xl">Prerequisites</Title>
      <Text size="md" mb="md">
        This exercise uses your math-docs repository from Module 1, Exercise 2:
        <code>git@github.com:your-username/math-docs.git</code>
      </Text>

      <Title order={2} mb="md" mt="xl">Step 1: Create Package with CLI</Title>

      <Text size="md" mb="md">
        First, let's understand how to build a package with cli.
      </Text>

      <Title order={3} mb="sm">Clone and Setup</Title>
      <CodeBlock
        code={`git clone git@github.com:your-username/math-docs.git
cd math-docs
git checkout -b testing-exercise`}
        language="bash"
      />

      <Title order={3} mb="sm" mt="lg">Simple Package Structure</Title>
      <Text size="md" mb="md">
        Create a minimal package with command-line functionality:
      </Text>
      <CodeBlock
        code={`mathtools/
├── pyproject.toml
├── mathtools/
│   ├── __init__.py
│   ├── calculator.py
│   └── cli.py
└── tests/
    ├── __init__.py
    └── test_calculator.py`}
      />

      <Title order={3} mb="sm" mt="lg">Package Configuration</Title>
      <CodeBlock
        code={`[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mathtools"
version = "0.1.0"
description = "Simple calculator with CLI"
requires-python = ">=3.8"
dependencies = []

[project.optional-dependencies]
test = ["pytest>=6.0"]

[project.scripts]
mathcalc = "mathtools.cli:main"`}
        language="toml"
      />

      <Title order={3} mb="sm" mt="lg">Core Calculator Functions</Title>
      <Text size="md" mb="md">
        Create <code>mathtools/calculator.py</code> with basic operations:
      </Text>
      <CodeBlock
        code={`def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Command-Line Interface</Title>
      <Text size="md" mb="md">
        Create <code>mathtools/cli.py</code> to use your functions from the command line:
      </Text>
      <CodeBlock
        code={`import sys
from .calculator import add, multiply

def main():
    if len(sys.argv) != 4:
        print("Usage: mathcalc <operation> <num1> <num2>")
        print("Operations: add, multiply")
        return

    op, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

    if op == "add":
        result = add(a, b)
    elif op == "multiply":
        result = multiply(a, b)
    else:
        print("Unknown operation")
        return

    print(f"{a} {op} {b} = {result}")`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Test Your Package</Title>
      <Text size="md" mb="md">
        Install and test the CLI functionality:
      </Text>
      <CodeBlock
        code={`cd mathtools
pip install -e .
mathcalc add 5 3
mathcalc multiply 4 7`}
        language="bash"
      />

      <Text size="md" mb="md">
        <strong>What did we accomplish?</strong> You now have a working Python package
        that can be installed and used from the command line. But how do we ensure
        it works correctly as we make changes?
      </Text>

      <Title order={2} mb="md" mt="xl">Step 2: Add Unit Tests</Title>

      <Text size="md" mb="md">
        Unit tests automatically verify that your functions work correctly.
        They catch bugs before users encounter them.
      </Text>

      <Title order={3} mb="sm">Create Basic Tests</Title>
      <Text size="md" mb="md">
        Create <code>tests/test_calculator.py</code>:
      </Text>
      <CodeBlock
        code={`from mathtools.calculator import add, multiply

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Install Test Dependencies</Title>
      <CodeBlock
        code={`pip install -e .[test]`}
        language="bash"
      />

      <Title order={3} mb="sm" mt="lg">Run Tests</Title>
      <CodeBlock
        code={`pytest tests/ -v`}
        language="bash"
      />

      <Text size="md" mb="md">
        <strong>What happens when you run tests?</strong> Pytest automatically finds
        and runs all test functions. If any test fails, you'll see exactly which
        function is broken and why.
      </Text>

      <Title order={3} mb="sm" mt="lg">Add More Test Cases</Title>
      <Text size="md" mb="md">
        Add edge cases to catch potential bugs:
      </Text>
      <CodeBlock
        code={`def test_add_edge_cases():
    assert add(0, 0) == 0
    assert add(-5, -3) == -8
    assert add(1.5, 2.5) == 4.0

def test_multiply_edge_cases():
    assert multiply(-2, 3) == -6
    assert multiply(2.5, 4) == 10.0`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Understand Test Benefits</Title>
      <List spacing="sm" size="md" mb="md">
        <List.Item><strong>Catch bugs early:</strong> Tests find problems before users do</List.Item>
        <List.Item><strong>Safe refactoring:</strong> Change code confidently knowing tests will catch breaks</List.Item>
        <List.Item><strong>Documentation:</strong> Tests show how your functions should be used</List.Item>
        <List.Item><strong>Regression prevention:</strong> Once fixed, bugs stay fixed</List.Item>
      </List>

      <Text size="md" mb="md">
        <strong>Try this:</strong> Intentionally break your <code>add</code> function
        (change <code>return a + b</code> to <code>return a - b</code>) and run the tests.
        See how quickly tests catch the error!
      </Text>

      <Title order={2} mb="md" mt="xl">Step 3: Continuous Integration (CI)</Title>

      <Text size="md" mb="md">
        CI automatically runs your tests every time you push code changes.
        This ensures your code always works, even when multiple people contribute.
      </Text>

      <Title order={3} mb="sm">Create GitHub Actions Workflow</Title>
      <Text size="md" mb="md">
        Create <code>.github/workflows/test.yml</code>:
      </Text>
      <CodeBlock
        code={`name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    - name: Install package
      run: |
        cd mathtools
        pip install -e .[test]
    - name: Run tests
      run: |
        cd mathtools
        pytest tests/ -v`}
        language="yaml"
      />

      <Title order={3} mb="sm" mt="lg">Push and Observe CI</Title>
      <CodeBlock
        code={`git add .
git commit -m "Add tests and CI workflow"
git push origin testing-exercise`}
        language="bash"
      />

      <Text size="md" mb="md">
        <strong>What happens next?</strong> Go to your GitHub repository and click the "Actions" tab.
        You'll see your workflow running automatically!
      </Text>

      <Title order={3} mb="sm" mt="lg">Understanding CI Benefits</Title>
      <List spacing="sm" size="md" mb="md">
        <List.Item><strong>Automatic validation:</strong> Every code change is tested immediately</List.Item>
        <List.Item><strong>Multiple environments:</strong> Tests run on clean systems, not just your machine</List.Item>
        <List.Item><strong>Team collaboration:</strong> Prevents broken code from being merged</List.Item>
        <List.Item><strong>Confidence:</strong> Green checkmarks mean your code is working</List.Item>
      </List>

      <Title order={3} mb="sm" mt="lg">Experiment with CI</Title>
      <Text size="md" mb="md">
        Try these experiments to see CI in action:
      </Text>
      <List spacing="sm" size="md" mb="md">
        <List.Item>Add a new function and test, push the changes</List.Item>
        <List.Item>Intentionally break a test, see the red X in GitHub</List.Item>
        <List.Item>Fix the test, watch it turn green again</List.Item>
      </List>

      <Alert icon={<IconAlertCircle />} color="blue" mt="md">
        <strong>Real-world impact:</strong> Companies use CI to test thousands of changes daily.
        Without it, software would be full of bugs and development would be much slower.
      </Alert>

      <Title order={2} mb="md" mt="xl">Why This Matters</Title>

      <Text size="md" mb="md">
        You've now experienced the development workflow used by professional software teams:
      </Text>

      <List spacing="sm" size="md" mb="md">
        <List.Item><strong>Write code</strong> → Create functionality</List.Item>
        <List.Item><strong>Write tests</strong> → Ensure it works correctly</List.Item>
        <List.Item><strong>Setup CI</strong> → Automatically validate all changes</List.Item>
        <List.Item><strong>Push changes</strong> → Get immediate feedback</List.Item>
      </List>

      <Text size="md" mb="md">
        This process prevents bugs, enables safe collaboration, and builds confidence in your code.
        It's the foundation of reliable software development.
      </Text>

      <Grid>
        <Grid.Col>
          <EvaluationModal module={2} />
        </Grid.Col>
      </Grid>
    </Container>
  );
};

export default Exercise2;