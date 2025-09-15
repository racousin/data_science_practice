import React from "react";
import { Container, Grid, Title, Text, List, Alert } from '@mantine/core';
import { IconAlertCircle } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";
import EvaluationModal from "components/EvaluationModal";

const Exercise3 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md">Exercise 3<span style={{color: 'red', fontWeight: 'bold'}}>*</span>: Package Distribution and Jupyter Integration</Title>

      <Text size="md" mb="md">
        Learn how to complete your package for distribution and integrate it with Jupyter notebooks
        for interactive data science workflows.
      </Text>

      <Title order={2} mb="md" mt="xl">Prerequisites</Title>
      <Text size="md" mb="md">
        This exercise continues with your mathtools package from Exercise 2:
        <code>git@github.com:your-username/math-docs.git</code>
      </Text>

      <Title order={2} mb="md" mt="xl">Step 1: Complete Package Configuration</Title>

      <Text size="md" mb="md">
        Let's complete your package with essential distribution files.
      </Text>

      <Title order={3} mb="sm">Switch to Your Testing Branch</Title>
      <CodeBlock
        code={`cd math-docs/mathtools
git checkout testing-exercise`}
        language="bash"
      />

      <Title order={3} mb="sm" mt="lg">Add .gitignore File</Title>
      <Text size="md" mb="md">
        Create <code>.gitignore</code> to exclude unnecessary files:
      </Text>
      <CodeBlock
        code={`__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.DS_Store`}
        language="text"
      />

      <Text size="md" mb="md">
        <strong>Why .gitignore?</strong> Prevents Python cache files and build artifacts
        from cluttering your repository.
      </Text>

      <Title order={3} mb="sm" mt="lg">Create requirements.txt</Title>
      <Text size="md" mb="md">
        Add <code>requirements.txt</code> for development dependencies:
      </Text>
      <CodeBlock
        code={`pytest>=6.0
jupyter>=1.0.0
notebook>=6.0.0`}
        language="text"
      />

      <Text size="md" mb="md">
        <strong>What's this for?</strong> Other developers can install all dependencies
        with <code>pip install -r requirements.txt</code>.
      </Text>

      <Title order={3} mb="sm" mt="lg">Update pyproject.toml</Title>
      <Text size="md" mb="md">
        Add Jupyter as an optional dependency in your existing <code>pyproject.toml</code>:
      </Text>
      <CodeBlock
        code={`[project.optional-dependencies]
test = ["pytest>=6.0"]
jupyter = ["jupyter>=1.0.0", "notebook>=6.0.0"]`}
        language="toml"
      />

      <Title order={2} mb="md" mt="xl">Step 2: Create a Jupyter Notebook</Title>

      <Text size="md" mb="md">
        Let's create a simple notebook that demonstrates your mathtools package.
      </Text>

      <Title order={3} mb="sm">Install Jupyter Dependencies</Title>
      <CodeBlock
        code={`pip install -e .[jupyter]`}
        language="bash"
      />

      <Title order={3} mb="sm" mt="lg">Create Demo Notebook</Title>
      <Text size="md" mb="md">
        Create <code>demo_notebook.ipynb</code> in your mathtools directory:
      </Text>
      <CodeBlock
        code={`jupyter notebook`}
        language="bash"
      />

      <Text size="md" mb="md">
        In the notebook, create these cells:
      </Text>

      <Title order={4} mb="sm" mt="md">Cell 1: Import and Setup</Title>
      <CodeBlock
        code={`# Demo of mathtools package
from mathtools.calculator import add, multiply
import matplotlib.pyplot as plt
import numpy as np

print("Testing mathtools package in Jupyter!")
print(f"5 + 3 = {add(5, 3)}")
print(f"4 * 7 = {multiply(4, 7)}")`}
        language="python"
      />

      <Title order={4} mb="sm" mt="md">Cell 2: Visualize Results</Title>
      <CodeBlock
        code={`# Create some data using our functions
x = np.arange(1, 11)
y_add = [add(i, 5) for i in x]
y_mult = [multiply(i, 2) for i in x]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y_add, 'b-o')
plt.title('Addition: x + 5')
plt.xlabel('x')
plt.ylabel('Result')

plt.subplot(1, 2, 2)
plt.plot(x, y_mult, 'r-s')
plt.title('Multiplication: x * 2')
plt.xlabel('x')
plt.ylabel('Result')
plt.tight_layout()
plt.show()`}
        language="python"
      />


      <Text size="md" mb="md">
        <strong>What did we create?</strong> A notebook that demonstrates your package
        with visualizations and interactive features.
      </Text>

      <Title order={2} mb="md" mt="xl">Step 3: Google Colab Integration</Title>

      <Text size="md" mb="md">
        Now let's make your package work in Google Colab.
      </Text>

      <Title order={3} mb="sm">Make Repository Public</Title>
      <Text size="md" mb="md">
        First, ensure your repository is public on GitHub:
      </Text>
      <List spacing="sm" size="md" mb="md">
        <List.Item>Go to your repository settings</List.Item>
        <List.Item>Scroll to "Danger Zone"</List.Item>
        <List.Item>Click "Change repository visibility" → "Make public"</List.Item>
      </List>

      <Title order={3} mb="sm" mt="lg">Commit Your Changes</Title>
      <CodeBlock
        code={`git add .
git commit -m "Add package distribution files and Jupyter demo"
git push origin testing-exercise`}
        language="bash"
      />

      <Title order={3} mb="sm" mt="lg">Save and Push Your Notebook</Title>
      <Text size="md" mb="md">
        Save your notebook as <code>colab_demo.ipynb</code> and push to GitHub:
      </Text>
      <CodeBlock
        code={`# Save notebook, then commit
git add colab_demo.ipynb
git commit -m "Add Colab demo notebook"
git push origin testing-exercise`}
        language="bash"
      />

      <Title order={3} mb="sm" mt="lg">Open Notebook in Colab</Title>
      <Text size="md" mb="md">
        Now you can open your notebook directly from GitHub in Colab:
      </Text>

      <Text size="md" mb="md">
        <strong>Method 1: Direct URL</strong><br/>
        Use this URL pattern (replace with your username):
      </Text>
      <CodeBlock
        code={`https://colab.research.google.com/github/your-username/math-docs/blob/testing-exercise/mathtools/colab_demo.ipynb`}
        language="text"
      />

      <Title order={3} mb="sm" mt="lg">Add Colab Badge to README</Title>
      <Text size="md" mb="md">
        Update your <code>mathtools/README.md</code> with a Colab badge:
      </Text>
      <CodeBlock
        code={`# Mathtools Package

Simple calculator package with CLI functionality.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/math-docs/blob/testing-exercise/mathtools/colab_demo.ipynb)

## Installation
\`\`\`bash
git clone https://github.com/your-username/math-docs.git
cd math-docs/mathtools
pip install -e .
\`\`\`

## Usage
\`\`\`bash
# Command line
mathcalc add 5 3
mathcalc multiply 4 7

# In Python
from mathtools.calculator import add, multiply
result = add(10, 15)
\`\`\`

## Demo
See the interactive demo in the Colab notebook above!`}
        language="markdown"
      />

      <Text size="md" mb="md">
        <strong>What's the Colab badge?</strong> The badge provides a direct link to open
        your notebook in Colab. Anyone can click it and immediately run your code!
      </Text>

      <Title order={3} mb="sm" mt="lg">Test Your Colab Integration</Title>
      <Text size="md" mb="md">
        In your Colab notebook, add this first cell to install your package:
      </Text>
      <CodeBlock
        code={`# Install mathtools package from GitHub
!git clone https://github.com/your-username/math-docs.git
!cd math-docs/mathtools && pip install -e .`}
        language="python"
      />

      <Text size="md" mb="md">
        <strong>Important:</strong> When someone opens your notebook in Colab, they'll need
        to run this installation cell first to use your package.
      </Text>

      <Title order={2} mb="md" mt="xl">Why This Workflow Matters</Title>

      <Text size="md" mb="md">
        You've now created a complete data science package workflow:
      </Text>

      <List spacing="sm" size="md" mb="md">
        <List.Item><strong>Local development:</strong> Write and test code locally</List.Item>
        <List.Item><strong>Package distribution:</strong> Share code as installable packages</List.Item>
        <List.Item><strong>Jupyter integration:</strong> Interactive development and demos</List.Item>
        <List.Item><strong>Cloud accessibility:</strong> Anyone can run your code in Colab</List.Item>
      </List>

      <Text size="md" mb="md">
        Your package can now be used by anyone, anywhere, without complex setup.
      </Text>

      <Grid>
        <Grid.Col>
          <EvaluationModal module={2} />
        </Grid.Col>
      </Grid>
    </Container>
  );
};

export default Exercise3;