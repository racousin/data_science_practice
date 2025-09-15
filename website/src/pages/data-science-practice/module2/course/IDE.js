import React from "react";
import { Container, Grid } from '@mantine/core';
import CodeBlock from "components/CodeBlock";

const Ide = () => {
  return (
    <Container fluid>
      <div data-slide>
        <h1>Integrated Development Environments (IDEs)</h1>
        <p>
          An Integrated Development Environment (IDE) is a software application that provides
          comprehensive facilities for software development. IDEs combine code editing, debugging,
          building, and version control in a single interface.
        </p>
      </div>

      <div data-slide>
        <h2>Why Use an IDE for Data Science?</h2>
        <p>IDEs provide several advantages for data science work:</p>
        <ul>
          <li><strong>Code completion:</strong> Intelligent suggestions as you type</li>
          <li><strong>Syntax highlighting:</strong> Color-coded syntax for better readability</li>
          <li><strong>Debugging tools:</strong> Step through code and inspect variables</li>
          <li><strong>Package management:</strong> Easy installation and management of libraries</li>
          <li><strong>Version control integration:</strong> Built-in Git support</li>
        </ul>
      </div>

      <div data-slide>
        <h2>Popular IDEs for Python Data Science</h2>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <h3>Visual Studio Code</h3>
            <p>
              A lightweight, extensible code editor with excellent Python support through extensions.
            </p>
            <ul>
              <li>Free and open-source</li>
              <li>Rich ecosystem of extensions</li>
              <li>Integrated terminal and Git support</li>
              <li>Jupyter notebook integration</li>
            </ul>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h2>PyCharm</h2>
        <p>
          A full-featured IDE specifically designed for Python development.
        </p>
        <ul>
          <li>Intelligent code completion and analysis</li>
          <li>Professional debugging and testing tools</li>
          <li>Built-in database tools</li>
          <li>Scientific tools integration (NumPy, Matplotlib, etc.)</li>
        </ul>
      </div>


      <div data-slide>
        <h2>Setting Up VS Code for Python</h2>
        <p>Follow these steps to configure VS Code for Python development:</p>
        <ol>
          <li>Download and install VS Code from the official website</li>
          <li>Install the Python extension by Microsoft</li>
          <li>Configure the Python interpreter</li>
        </ol>
        <CodeBlock code={`# Open VS Code and press Ctrl+Shift+P (Cmd+Shift+P on Mac)
# Type "Python: Select Interpreter"
# Choose your Python installation`} language="python" />
      </div>

      <div data-slide>
        <h2>Essential VS Code Extensions for Data Science</h2>
        <ul>
          <li><strong>Python:</strong> Core Python language support</li>
          <li><strong>Jupyter:</strong> Jupyter notebook support in VS Code</li>
          <li><strong>GitLens:</strong> Enhanced Git capabilities</li>
          <li><strong>Pylance:</strong> Fast, feature-rich Python language server</li>
        </ul>
      </div>

      <div data-slide>
        <h2>Creating Your First Python Project</h2>
        <p>Let's create a simple data science project structure:</p>
        <CodeBlock code={`mkdir my_data_project
cd my_data_project
mkdir data notebooks src tests`} language="bash" />
        <p>This creates a project with organized folders for:</p>
        <ul>
          <li><code>data/</code> - Raw and processed data files</li>
          <li><code>notebooks/</code> - Jupyter notebooks for exploration</li>
          <li><code>src/</code> - Python source code modules</li>
          <li><code>tests/</code> - Unit tests for your code</li>
        </ul>
      </div>

      <div data-slide>
        <h2>IDE Features for Data Science Workflow</h2>
        <p>Modern IDEs provide specialized features for data science:</p>
        <ul>
          <li><strong>Variable explorer:</strong> Inspect DataFrames and arrays</li>
          <li><strong>Interactive plotting:</strong> Visualize data directly in the IDE</li>
          <li><strong>Integrated terminal:</strong> Run commands without leaving the IDE</li>
          <li><strong>Code formatting:</strong> Automatic code styling with tools like Black</li>
          <li><strong>Linting:</strong> Real-time code quality checks</li>
        </ul>
      </div>

      <div data-slide>
        <h2>Best Practices</h2>
        <ul>
          <li>Use virtual environments to isolate project dependencies</li>
          <li>Configure your IDE to use the correct Python interpreter</li>
          <li>Set up code formatting and linting for consistent code quality</li>
          <li>Learn keyboard shortcuts to improve productivity</li>
          <li>Use version control integration for tracking changes</li>
        </ul>
      </div>
    </Container>
  );
};

export default Ide;