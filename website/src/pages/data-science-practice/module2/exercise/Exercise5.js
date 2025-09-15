import React from "react";
import { Container, Title, Text, List, Space, Alert } from '@mantine/core';
import { IconInfoCircle, IconCheck } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";

const Exercise5 = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">
        Exercise 5: CI/CD with GitHub Actions
      </Title>

      <Title order={2} mb="md">Scenario</Title>
      <Text size="md" mb="lg">
        Your data utils package from Exercise 4 is growing, and you want to ensure code quality.
        Set up automated testing and continuous integration to catch bugs before they reach users.
      </Text>

      <Alert icon={<IconInfoCircle />} color="blue" mb="lg">
        CI/CD (Continuous Integration/Continuous Deployment) automatically tests your code every time you make changes.
        This prevents broken code from being merged and gives you confidence in your releases.
      </Alert>

      <Title order={2} mb="md">Tasks</Title>

      <Title order={3} mt="lg" mb="sm">1. Add Tests to Your Package</Title>
      <Text size="md" mb="sm">Extend your data-utils-package with a tests directory:</Text>
      <CodeBlock
        code={`data-utils-package/
├── setup.py
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── datautils/
│   ├── __init__.py
│   ├── cleaner.py
│   └── analyzer.py
└── tests/
    ├── __init__.py
    ├── test_cleaner.py
    └── test_analyzer.py`}
        language="text"
      />

      <Title order={3} mt="lg" mb="sm">2. Write Tests for Cleaner Functions</Title>
      <Text size="md" mb="sm">Create <code>tests/test_cleaner.py</code>:</Text>
      <CodeBlock
        code={`import pytest
import pandas as pd
import numpy as np
from datautils.cleaner import remove_outliers, fill_missing_values, normalize_data

def test_remove_outliers():
    """Test outlier removal function."""
    # Create data with clear outliers
    data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
    result = remove_outliers(data, threshold=2)

    # Should remove the outlier
    assert len(result) < len(data)
    assert 100 not in result.values

def test_fill_missing_values_mean():
    """Test filling missing values with mean."""
    data = pd.Series([1, 2, np.nan, 4, 5])
    result = fill_missing_values(data, method='mean')

    # Should have no missing values
    assert result.isnull().sum() == 0
    # Missing value should be replaced with mean (3.0)
    assert result.iloc[2] == 3.0

def test_fill_missing_values_median():
    """Test filling missing values with median."""
    data = pd.Series([1, 2, np.nan, 4, 5])
    result = fill_missing_values(data, method='median')

    assert result.isnull().sum() == 0
    assert result.iloc[2] == 3.0  # median of [1,2,4,5]

def test_normalize_data():
    """Test data normalization."""
    data = pd.Series([10, 20, 30, 40, 50])
    result = normalize_data(data)

    # Should be normalized to 0-1 range
    assert result.min() == 0.0
    assert result.max() == 1.0
    assert len(result) == len(data)`}
        language="python"
      />

      <Title order={3} mt="lg" mb="sm">3. Write Tests for Analyzer Functions</Title>
      <Text size="md" mb="sm">Create <code>tests/test_analyzer.py</code>:</Text>
      <CodeBlock
        code={`import pytest
import pandas as pd
import numpy as np
from datautils.analyzer import quick_stats, correlation_summary, missing_data_report

def test_quick_stats():
    """Test quick statistics function."""
    data = pd.Series([1, 2, 3, 4, 5])
    result = quick_stats(data)

    # Check all expected keys exist
    expected_keys = ['count', 'mean', 'median', 'std', 'min', 'max']
    assert all(key in result for key in expected_keys)

    # Check specific values
    assert result['count'] == 5
    assert result['mean'] == 3.0
    assert result['median'] == 3.0
    assert result['min'] == 1
    assert result['max'] == 5

def test_correlation_summary():
    """Test correlation summary function."""
    # Create simple correlated data
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],  # perfectly correlated
        'target': [1, 2, 3, 4, 5]
    })

    result = correlation_summary(df, 'target')

    # Should return correlations (excluding target itself)
    assert 'target' not in result.index
    assert 'feature1' in result.index
    assert 'feature2' in result.index

def test_missing_data_report():
    """Test missing data report function."""
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4],
        'col2': [1, np.nan, np.nan, 4],
        'col3': [1, 2, 3, 4]
    })

    result = missing_data_report(df)

    # Check structure
    assert 'missing_count' in result
    assert 'missing_percent' in result

    # Check values
    assert result['missing_count']['col1'] == 1
    assert result['missing_count']['col2'] == 2
    assert result['missing_count']['col3'] == 0
    assert result['missing_percent']['col2'] == 50.0`}
        language="python"
      />

      <Title order={3} mt="lg" mb="sm">4. Update Setup.py with Test Dependencies</Title>
      <Text size="md" mb="sm">Add test dependencies to your <code>setup.py</code>:</Text>
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
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ]
    },
    python_requires=">=3.7",
)`}
        language="python"
      />

      <Title order={3} mt="lg" mb="sm">5. Create GitHub Actions Workflow</Title>
      <Text size="md" mb="sm">Create <code>.github/workflows/ci.yml</code>:</Text>
      <CodeBlock
        code={`name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python \${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: \${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run tests
      run: |
        pytest tests/ -v --cov=datautils --cov-report=term-missing

    - name: Check code quality
      run: |
        # Install flake8 for basic code quality check
        pip install flake8
        flake8 datautils/ --max-line-length=88 --ignore=E203,W503`}
        language="yaml"
      />

      <Title order={3} mt="lg" mb="sm">6. Add Test Configuration</Title>
      <Text size="md" mb="sm">Create <code>pytest.ini</code> for test configuration:</Text>
      <CodeBlock
        code={`[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short`}
        language="ini"
      />

      <Title order={3} mt="lg" mb="sm">7. Test Locally First</Title>
      <Text size="md" mb="sm">Before pushing, test everything locally:</Text>
      <CodeBlock
        code={`# Install with dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=datautils --cov-report=html

# Check code style
pip install flake8
flake8 datautils/ --max-line-length=88`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">8. Commit and Push</Title>
      <Text size="md" mb="sm">Add all new files and push to trigger CI:</Text>
      <CodeBlock
        code={`git add .
git commit -m "Add comprehensive testing and CI/CD pipeline"
git push origin main`}
        language="bash"
      />

      <Title order={3} mt="lg" mb="sm">9. Monitor GitHub Actions</Title>
      <Text size="md" mb="sm">Check your CI pipeline:</Text>
      <List spacing="sm">
        <List.Item>Go to your repository on GitHub</List.Item>
        <List.Item>Click the "Actions" tab</List.Item>
        <List.Item>Watch your workflow run</List.Item>
        <List.Item>Check for green checkmarks (passing tests)</List.Item>
        <List.Item>If red X appears, click to see error details</List.Item>
      </List>

      <Title order={3} mt="lg" mb="sm">10. Create a Pull Request</Title>
      <Text size="md" mb="sm">Test your CI with a pull request:</Text>
      <CodeBlock
        code={`# Create a new branch
git checkout -b improve-documentation

# Make a small change to README.md
echo "## Latest Updates" >> README.md
echo "- Added comprehensive testing suite" >> README.md
echo "- Implemented CI/CD with GitHub Actions" >> README.md

# Commit and push
git add README.md
git commit -m "Update documentation with testing info"
git push origin improve-documentation`}
        language="bash"
      />
      <Text size="md" mb="sm">Create a pull request on GitHub and watch the CI run automatically!</Text>

      <Title order={2} mt="xl" mb="md">Understanding the CI Pipeline</Title>
      <Text size="md" mb="sm">Your GitHub Actions workflow does the following:</Text>
      <List spacing="sm">
        <List.Item><strong>Triggers:</strong> Runs on every push to main and every pull request</List.Item>
        <List.Item><strong>Matrix Testing:</strong> Tests against Python 3.8, 3.9, and 3.10</List.Item>
        <List.Item><strong>Dependencies:</strong> Installs your package with dev dependencies</List.Item>
        <List.Item><strong>Testing:</strong> Runs all tests with coverage reporting</List.Item>
        <List.Item><strong>Quality Check:</strong> Checks code style with flake8</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md">Expected Result</Title>
      <Text size="md" mb="sm">Your completed CI/CD setup should have:</Text>
      <List spacing="xs">
        <List.Item icon={<IconCheck />}>Comprehensive test suite covering all functions</List.Item>
        <List.Item icon={<IconCheck />}>GitHub Actions workflow that runs automatically</List.Item>
        <List.Item icon={<IconCheck />}>Tests running on multiple Python versions</List.Item>
        <List.Item icon={<IconCheck />}>Code coverage reporting</List.Item>
        <List.Item icon={<IconCheck />}>Basic code quality checks</List.Item>
        <List.Item icon={<IconCheck />}>Green badges showing passing tests</List.Item>
      </List>

      <Alert icon={<IconInfoCircle />} color="green" mt="lg">
        <strong>Why This Matters:</strong> With CI/CD, you catch bugs before they reach users.
        Every code change is automatically tested, ensuring your package remains reliable.
        This builds trust with users and makes collaboration safer.
      </Alert>

      <Title order={2} mt="xl" mb="md">Bonus: Add Status Badge</Title>
      <Text size="md" mb="sm">Add a CI status badge to your README.md:</Text>
      <CodeBlock
        code={`# Data Utils Package

![CI](https://github.com/your-username/data-utils-package/workflows/CI/badge.svg)

A simple Python package for common data science utilities with automated testing.`}
        language="markdown"
      />

      <Space h="xl" />
    </Container>
  );
};

export default Exercise5;