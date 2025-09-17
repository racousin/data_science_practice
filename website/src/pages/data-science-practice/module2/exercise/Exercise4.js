import React from "react";
import { Container, Grid, Title, Text, List, Alert } from '@mantine/core';
import { IconAlertCircle } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";
import EvaluationModal from "components/EvaluationModal";

const Exercise4 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md">Exercise 4: Data Science Workflow Integration</Title>

      <Text size="md" mb="md">
        Extend your mathtools package with basic data science capabilities and learn how
        packages integrate into real-world data workflows.
      </Text>

      <Title order={2} mb="md" mt="xl">Prerequisites</Title>
      <Text size="md" mb="md">
        This exercise continues with your mathtools package from Exercise 3:
        <code>git@github.com:your-username/math-docs.git</code>
      </Text>

      <Title order={2} mb="md" mt="xl">Step 1: Add Data Science Dependencies</Title>

      <Text size="md" mb="md">
        Let's extend your package to handle basic data operations that are common in data science.
      </Text>

      <Title order={3} mb="sm">Update Package Dependencies</Title>
      <Text size="md" mb="md">
        First, update your <code>pyproject.toml</code> to include data science libraries:
      </Text>
      <CodeBlock
        code={`[project]
name = "mathtools"
version = "0.2.0"
description = "Calculator with data science utilities"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "matplotlib>=3.5.0"
]

[project.optional-dependencies]
test = ["pytest>=6.0"]
jupyter = ["jupyter>=1.0.0", "notebook>=6.0.0"]
dev = ["pytest>=6.0", "jupyter>=1.0.0", "notebook>=6.0.0"]`}
        language="toml"
      />

      <Text size="md" mb="md">
        <strong>Why these libraries?</strong> NumPy for numerical operations, Pandas for data
        handling, and Matplotlib for basic visualization - the foundation of data science.
      </Text>

      <Title order={3} mb="sm" mt="lg">Update requirements.txt</Title>
      <CodeBlock
        code={`pytest>=6.0
jupyter>=1.0.0
notebook>=6.0.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.5.0`}
        language="text"
      />

      <Title order={2} mb="md" mt="xl">Step 2: Add Data Processing Module</Title>

      <Text size="md" mb="md">
        Create a new module for basic data operations.
      </Text>

      <Title order={3} mb="sm">Create data_utils.py</Title>
      <Text size="md" mb="md">
        Add <code>mathtools/data_utils.py</code> with simple data functions:
      </Text>
      <CodeBlock
        code={`import numpy as np
import pandas as pd

def create_sample_data(n=100):
    """Generate sample dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.normal(10, 2, n),
        'y': np.random.normal(20, 3, n),
        'category': np.random.choice(['A', 'B', 'C'], n)
    })

def basic_stats(data):
    """Calculate basic statistics."""
    if isinstance(data, pd.Series):
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max()
        }
    return data.describe().to_dict()`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Add Simple Analysis Functions</Title>
      <Text size="md" mb="md">
        Extend <code>data_utils.py</code> with analysis capabilities:
      </Text>
      <CodeBlock
        code={`def correlation_analysis(df, col1, col2):
    """Calculate correlation between two columns."""
    correlation = df[col1].corr(df[col2])
    return {
        'correlation': correlation,
        'strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
    }

def group_summary(df, group_col, value_col):
    """Summarize data by groups."""
    return df.groupby(group_col)[value_col].agg(['mean', 'count', 'std']).round(2)`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Create Visualization Module</Title>
      <Text size="md" mb="md">
        Add <code>mathtools/visualize.py</code> for basic plots:
      </Text>
      <CodeBlock
        code={`import matplotlib.pyplot as plt

def simple_plot(data, x_col, y_col, title="Data Plot"):
    """Create a simple scatter plot."""
    plt.figure(figsize=(8, 6))
    plt.scatter(data[x_col], data[y_col], alpha=0.6)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt

def distribution_plot(data, column, title="Distribution"):
    """Create a histogram."""
    plt.figure(figsize=(8, 6))
    plt.hist(data[column], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Update Package Imports</Title>
      <Text size="md" mb="md">
        Update <code>mathtools/__init__.py</code> to include new modules:
      </Text>
      <CodeBlock
        code={`from .calculator import add, multiply
from .data_utils import create_sample_data, basic_stats, correlation_analysis, group_summary
from .visualize import simple_plot, distribution_plot

__version__ = "0.2.0"
__author__ = "Your Name"`}
        language="python"
      />

      <Title order={2} mb="md" mt="xl">Step 3: Create Data Science Workflow Example</Title>

      <Text size="md" mb="md">
        Let's create a complete example showing how your package fits into a data workflow.
      </Text>

      <Title order={3} mb="sm">Update Your Jupyter Notebook</Title>
      <Text size="md" mb="md">
        Replace your <code>colab_demo.ipynb</code> with a data science workflow example:
      </Text>

      <Title order={4} mb="sm" mt="md">Cell 1: Setup and Data Creation</Title>
      <CodeBlock
        code={`# Data Science Workflow with mathtools
import mathtools
import matplotlib.pyplot as plt

# Create sample dataset
df = mathtools.create_sample_data(200)
print("Dataset created with shape:", df.shape)
print("\\nFirst 5 rows:")
df.head()`}
        language="python"
      />

      <Title order={4} mb="sm" mt="md">Cell 2: Basic Analysis</Title>
      <CodeBlock
        code={`# Basic statistical analysis
print("=== Basic Statistics ===")
x_stats = mathtools.basic_stats(df['x'])
y_stats = mathtools.basic_stats(df['y'])

print("X column stats:", x_stats)
print("Y column stats:", y_stats)

# Correlation analysis
corr_result = mathtools.correlation_analysis(df, 'x', 'y')
print(f"\\nCorrelation: {corr_result['correlation']:.3f} ({corr_result['strength']})")`}
        language="python"
      />

      <Title order={4} mb="sm" mt="md">Cell 3: Group Analysis</Title>
      <CodeBlock
        code={`# Group-based analysis
print("=== Analysis by Category ===")
group_stats = mathtools.group_summary(df, 'category', 'x')
print(group_stats)

# Use our calculator functions for custom metrics
total_observations = mathtools.add(len(df[df['category']=='A']),
                                  mathtools.add(len(df[df['category']=='B']),
                                               len(df[df['category']=='C'])))
print(f"\\nTotal observations: {total_observations}")`}
        language="python"
      />

      <Title order={4} mb="sm" mt="md">Cell 4: Visualization</Title>
      <CodeBlock
        code={`# Create visualizations
fig1 = mathtools.simple_plot(df, 'x', 'y', 'X vs Y Relationship')
plt.show()

fig2 = mathtools.distribution_plot(df, 'x', 'Distribution of X values')
plt.show()

print("Data science workflow complete!")`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Add Tests for New Functions</Title>
      <Text size="md" mb="md">
        Extend <code>tests/test_calculator.py</code> to include data function tests:
      </Text>
      <CodeBlock
        code={`import pandas as pd
from mathtools.data_utils import create_sample_data, basic_stats

def test_sample_data_creation():
    df = create_sample_data(50)
    assert len(df) == 50
    assert list(df.columns) == ['x', 'y', 'category']

def test_basic_stats():
    data = pd.Series([1, 2, 3, 4, 5])
    stats = basic_stats(data)
    assert stats['mean'] == 3.0
    assert stats['median'] == 3.0`}
        language="python"
      />

      <Title order={2} mb="md" mt="xl">Step 4: Update Documentation and Deploy</Title>

      <Title order={3} mb="sm">Install and Test</Title>
      <CodeBlock
        code={`pip install -e .[dev]
pytest tests/ -v`}
        language="bash"
      />

      <Title order={3} mb="sm" mt="lg">Update README with Data Science Features</Title>
      <CodeBlock
        code={`# Mathtools Package

Calculator with data science utilities for basic workflows.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/math-docs/blob/testing-exercise/mathtools/colab_demo.ipynb)

## Features

### Basic Calculator
- Addition and multiplication functions
- Command-line interface

### Data Science Utilities
- Sample data generation
- Basic statistical analysis
- Correlation analysis
- Group summaries
- Simple visualizations

## Installation
\`\`\`bash
git clone https://github.com/your-username/math-docs.git
cd math-docs/mathtools
pip install -e .
\`\`\`

## Quick Start
\`\`\`python
import mathtools

# Create sample data
df = mathtools.create_sample_data(100)

# Analyze data
stats = mathtools.basic_stats(df['x'])
correlation = mathtools.correlation_analysis(df, 'x', 'y')

# Visualize
plot = mathtools.simple_plot(df, 'x', 'y')
\`\`\``}
        language="markdown"
      />

      <Title order={3} mb="sm" mt="lg">Commit Your Data Science Extension</Title>
      <CodeBlock
        code={`git add .
git commit -m "Add data science workflow capabilities"
git push origin testing-exercise`}
        language="bash"
      />

      <Title order={2} mb="md" mt="xl">Understanding the Data Science Package Pattern</Title>

      <Text size="md" mb="md">
        You've now created a package that follows common data science patterns:
      </Text>

      <List spacing="sm" size="md" mb="md">
        <List.Item><strong>Modular design:</strong> Separate modules for different functionality</List.Item>
        <List.Item><strong>Data generation:</strong> Built-in sample data for testing and demos</List.Item>
        <List.Item><strong>Analysis functions:</strong> Reusable statistical operations</List.Item>
        <List.Item><strong>Visualization helpers:</strong> Simple plotting capabilities</List.Item>
        <List.Item><strong>Workflow integration:</strong> Functions that work together seamlessly</List.Item>
      </List>

      <Alert icon={<IconAlertCircle />} color="blue" mt="md">
        <strong>Looking ahead:</strong> In the next module, you'll dive deep into data science
        techniques. This package structure gives you a foundation for organizing data science
        code as reusable, testable components rather than scattered notebook cells.
      </Alert>

      <Title order={2} mb="md" mt="xl">Real-World Applications</Title>

      <Text size="md" mb="md">
        This exercise demonstrates how data scientists structure their work:
      </Text>

      <List spacing="sm" size="md" mb="md">
        <List.Item><strong>Package organization:</strong> Keep related functions together</List.Item>
        <List.Item><strong>Dependency management:</strong> Clearly specify what libraries you need</List.Item>
        <List.Item><strong>Testing data functions:</strong> Ensure your analysis code works correctly</List.Item>
        <List.Item><strong>Documentation:</strong> Make your tools usable by others</List.Item>
        <List.Item><strong>Workflow reproducibility:</strong> Anyone can install and run your analysis</List.Item>
      </List>

      <Text size="md" mb="md">
        You're now ready to tackle complex data science projects with proper software engineering practices!
      </Text>

      <Grid>
        <Grid.Col>
          <EvaluationModal module={2} />
        </Grid.Col>
      </Grid>
    </Container>
  );
};

export default Exercise4;