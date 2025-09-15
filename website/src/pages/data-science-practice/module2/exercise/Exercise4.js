import React from "react";
import { Container, Title, Text, List, Space, Alert } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";

const Exercise4 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">
        Exercise 4: Package Creation and Google Colab Integration
      </Title>

      <Title order={2} mb="md">Scenario</Title>
      <Text size="md" mb="lg">
        You're working on a data science project and realize you're repeatedly writing the same utility functions.
        Instead of copying code between notebooks, create a reusable Python package and use it in Google Colab.
      </Text>

      <Alert icon={<IconInfoCircle />} color="blue" mb="lg">
        This exercise demonstrates why creating packages is better than writing all code in notebooks.
        You'll create a simple but useful data utilities package and import it into Colab.
      </Alert>

      <Title order={2} mb="md">Tasks</Title>

      <Title order={3} mt="lg" mb="sm">1. Repository Setup</Title>
      <Text size="md" mb="sm">Create a new repository for your package:</Text>
      <CodeBlock
        code={`# Create repository on GitHub: data-utils-package
git clone https://github.com/your-username/data-utils-package.git
cd data-utils-package`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">2. Create the Package Structure</Title>
      <Text size="md" mb="sm">Set up the following directory structure:</Text>
      <CodeBlock
        code={`data-utils-package/
├── setup.py
├── README.md
└── datautils/
    ├── __init__.py
    ├── cleaner.py
    └── analyzer.py`}
        language="text"
      />

      <Title order={3} mt="lg" mb="sm">3. Implement Data Cleaning Functions</Title>
      <Text size="md" mb="sm">Create <code>datautils/cleaner.py</code>:</Text>
      <CodeBlock
        code={`def remove_outliers(data, threshold=2):
    """Remove outliers using z-score method."""
    import numpy as np
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores < threshold]

def fill_missing_values(data, method='mean'):
    """Fill missing values with mean, median, or mode."""
    if method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    elif method == 'mode':
        return data.fillna(data.mode()[0])

def normalize_data(data):
    """Normalize data to 0-1 range."""
    return (data - data.min()) / (data.max() - data.min())`}
        language="python"
      />

      <Title order={3} mt="lg" mb="sm">4. Implement Data Analysis Functions</Title>
      <Text size="md" mb="sm">Create <code>datautils/analyzer.py</code>:</Text>
      <CodeBlock
        code={`def quick_stats(data):
    """Get quick statistics summary."""
    return {
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max()
    }

def correlation_summary(df, target_column):
    """Get correlation with target column."""
    correlations = df.corr()[target_column].sort_values(ascending=False)
    return correlations.drop(target_column)

def missing_data_report(df):
    """Generate missing data report."""
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    return {
        'missing_count': missing,
        'missing_percent': missing_percent
    }`}
        language="python"
      />

      <Title order={3} mt="lg" mb="sm">5. Configure Package Imports</Title>
      <Text size="md" mb="sm">Create <code>datautils/__init__.py</code>:</Text>
      <CodeBlock
        code={`from .cleaner import remove_outliers, fill_missing_values, normalize_data
from .analyzer import quick_stats, correlation_summary, missing_data_report

__version__ = "0.1.0"
__author__ = "Your Name"`}
        language="python"
      />

      <Title order={3} mt="lg" mb="sm">6. Create Setup Configuration</Title>
      <Text size="md" mb="sm">Create <code>setup.py</code>:</Text>
      <CodeBlock
        code={`from setuptools import setup, find_packages

setup(
    name="datautils",
    version="0.1.0",
    author="Your Name",
    description="Simple data utilities for data science projects",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.7",
)`}
        language="python"
      />

      <Title order={3} mt="lg" mb="sm">7. Document Your Package</Title>
      <Text size="md" mb="sm">Create <code>README.md</code>:</Text>
      <CodeBlock
        code={`# Data Utils Package

A simple Python package for common data science utilities.

## Installation

\`\`\`bash
pip install git+https://github.com/your-username/data-utils-package.git
\`\`\`

## Usage

\`\`\`python
import datautils

# Clean data
clean_data = datautils.remove_outliers(your_data)
filled_data = datautils.fill_missing_values(your_df)

# Analyze data
stats = datautils.quick_stats(your_data)
missing_report = datautils.missing_data_report(your_df)
\`\`\`

## Functions

### Data Cleaning
- \`remove_outliers(data, threshold=2)\`: Remove outliers using z-score
- \`fill_missing_values(data, method='mean')\`: Fill missing values
- \`normalize_data(data)\`: Normalize data to 0-1 range

### Data Analysis
- \`quick_stats(data)\`: Get summary statistics
- \`correlation_summary(df, target)\`: Get correlations with target
- \`missing_data_report(df)\`: Generate missing data report`}
        language="markdown"
      />

      <Title order={3} mt="lg" mb="sm">8. Commit and Push</Title>
      <Text size="md" mb="sm">Save your work to GitHub:</Text>
      <CodeBlock
        code={`git add .
git commit -m "Initial data utils package implementation"
git push origin main`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">9. Create Google Colab Notebook</Title>
      <Text size="md" mb="sm">
        Create a new Colab notebook to demonstrate your package.
        Copy this template to get started:
      </Text>
      <CodeBlock
        code={`# Data Utils Package Demo

## Install the package
!pip install git+https://github.com/your-username/data-utils-package.git

## Import libraries
import pandas as pd
import numpy as np
import datautils

## Create sample data
np.random.seed(42)
data = {
    'feature1': np.random.normal(100, 15, 1000),
    'feature2': np.random.normal(50, 10, 1000),
    'target': np.random.normal(75, 20, 1000)
}
df = pd.DataFrame(data)

# Add some missing values and outliers
df.loc[np.random.choice(df.index, 50), 'feature1'] = np.nan
df.loc[np.random.choice(df.index, 10), 'feature2'] = 999  # outliers

## Demonstrate your package
print("=== Missing Data Report ===")
missing_report = datautils.missing_data_report(df)
print(missing_report)

print("\\n=== Quick Stats (before cleaning) ===")
stats_before = datautils.quick_stats(df['feature2'])
print(stats_before)

print("\\n=== Data Cleaning ===")
# Fill missing values
df_clean = df.copy()
df_clean['feature1'] = datautils.fill_missing_values(df_clean['feature1'])

# Remove outliers from feature2
df_clean['feature2'] = datautils.remove_outliers(df_clean['feature2'])

print("\\n=== Quick Stats (after cleaning) ===")
stats_after = datautils.quick_stats(df_clean['feature2'])
print(stats_after)

print("\\n=== Correlation Analysis ===")
correlations = datautils.correlation_summary(df_clean, 'target')
print(correlations)`}
        language="python"
      />

      <Title order={3} mt="lg" mb="sm">10. Save and Share Colab</Title>
      <Text size="md" mb="sm">Instructions for your Colab notebook:</Text>
      <List spacing="sm">
        <List.Item>Save the notebook to your Google Drive</List.Item>
        <List.Item>Set sharing permissions to "Anyone with the link can view"</List.Item>
        <List.Item>Copy the Colab sharing link</List.Item>
        <List.Item>Update your GitHub repository README with the Colab link</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md">Expected Result</Title>
      <Text size="md" mb="sm">Your completed exercise should have:</Text>
      <List spacing="xs">
        <List.Item>A working Python package hosted on GitHub</List.Item>
        <List.Item>Package installable via pip from GitHub</List.Item>
        <List.Item>Google Colab notebook demonstrating package usage</List.Item>
        <List.Item>Clear documentation in README</List.Item>
        <List.Item>Reusable functions that solve common data science problems</List.Item>
      </List>

      <Alert icon={<IconInfoCircle />} color="green" mt="lg">
        <strong>Why This Matters:</strong> Instead of copying the same functions across multiple notebooks,
        you now have a reusable package. Other data scientists can install and use your utilities,
        and you can continuously improve the package without updating every notebook.
      </Alert>

      <Space h="xl" />
    </Container>
  );
};

export default Exercise4;