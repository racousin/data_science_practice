import React from "react";
import { Container, Grid } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const SyntaxAndLinting = () => {
  return (
    <Container fluid>
      <div data-slide>
        <h2>Python Syntax and Code Formatting</h2>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <h4>Why Code Formatting Matters</h4>
            <p>
              Consistent code formatting improves readability, reduces errors, and makes
              collaboration easier. Python has established conventions documented in PEP 8
              that define how Python code should be formatted.
            </p>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h3>Introduction to Black</h3>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <p>
              Black is an uncompromising Python code formatter that automatically formats
              your code to follow consistent style guidelines. It removes debates about
              formatting by enforcing a single, deterministic style.
            </p>
            <h4>Installing Black</h4>
            <CodeBlock
              code={`pip install black`}
              language="bash"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h3>Using Black</h3>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <h4>Format a Single File</h4>
            <CodeBlock
              code={`black my_script.py`}
              language="bash"
            />

            <h4>Format All Python Files in Directory</h4>
            <CodeBlock
              code={`black .`}
              language="bash"
            />

            <h4>Check Without Making Changes</h4>
            <CodeBlock
              code={`black --check .`}
              language="bash"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h3>Introduction to Linting</h3>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <p>
              Linting analyzes code for potential errors, bugs, stylistic problems,
              and suspicious constructs. It helps catch issues before code execution
              and enforces coding standards.
            </p>
            <h4>Popular Python Linters</h4>
            <ul>
              <li><strong>flake8:</strong> Combines PyFlakes, pycodestyle, and McCabe complexity checker</li>
              <li><strong>pylint:</strong> Comprehensive linter with detailed error reporting</li>
              <li><strong>ruff:</strong> Fast linter written in Rust</li>
            </ul>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h3>Using Flake8</h3>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <h4>Installation</h4>
            <CodeBlock
              code={`pip install flake8`}
              language="bash"
            />

            <h4>Basic Usage</h4>
            <CodeBlock
              code={`flake8 my_script.py`}
              language="bash"
            />

            <h4>Lint All Files</h4>
            <CodeBlock
              code={`flake8 .`}
              language="bash"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h3>Configuration Files</h3>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <p>
              Both Black and linters can be configured using configuration files
              to customize their behavior for your project.
            </p>

            <h4>pyproject.toml Example</h4>
            <CodeBlock
              code={`[tool.black]
line-length = 88
target-version = ['py38']

[tool.flake8]
max-line-length = 88
extend-ignore = E203, W503`}
              language="toml"
            />
          </Grid.Col>
        </Grid>
      </div>
    </Container>
  );
};

export default SyntaxAndLinting;