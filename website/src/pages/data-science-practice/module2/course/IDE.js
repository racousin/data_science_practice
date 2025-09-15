import React from "react";
import { Container, Grid, Title, Text, List, Anchor } from '@mantine/core';
import CodeBlock from "components/CodeBlock";

const Ide = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={1} mb="md">Integrated Development Environments (IDEs)</Title>
        <Text size="md" mb="md">
          An Integrated Development Environment (IDE) is a software application that provides
          comprehensive facilities for software development. IDEs combine code editing, debugging,
          building, and version control in a single interface.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Why Use an IDE for Data Science?</Title>
        <Text size="md" mb="md">IDEs provide several advantages for data science work:</Text>
        <List spacing="sm" mb="md">
          <List.Item><Text fw={500} span>Code completion:</Text> Intelligent suggestions as you type</List.Item>
          <List.Item><Text fw={500} span>Syntax highlighting:</Text> Color-coded syntax for better readability</List.Item>
          <List.Item><Text fw={500} span>Debugging tools:</Text> Step through code and inspect variables</List.Item>
          <List.Item><Text fw={500} span>Package management:</Text> Easy installation and management of libraries</List.Item>
          <List.Item><Text fw={500} span>Version control integration:</Text> Built-in Git support</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Popular IDEs for Python Data Science</Title>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <Title order={3} mb="sm">
              Visual Studio Code{" "}
              <Anchor href="https://code.visualstudio.com/" target="_blank" rel="noopener noreferrer">
                (Website)
              </Anchor>
            </Title>
            <Text size="md" mb="md">
              A lightweight, extensible code editor with excellent Python support through extensions.
            </Text>
            <List spacing="sm" mb="md">
              <List.Item>Free and open-source</List.Item>
              <List.Item>Rich ecosystem of extensions</List.Item>
              <List.Item>Integrated terminal and Git support</List.Item>
              <List.Item>Jupyter notebook integration</List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={2} mb="md">
          PyCharm{" "}
          <Anchor href="https://www.jetbrains.com/pycharm/" target="_blank" rel="noopener noreferrer">
            (Website)
          </Anchor>
        </Title>
        <Text size="md" mb="md">
          A full-featured IDE specifically designed for Python development.
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Intelligent code completion and analysis</List.Item>
          <List.Item>Professional debugging and testing tools</List.Item>
          <List.Item>Built-in database tools</List.Item>
          <List.Item>Scientific tools integration (NumPy, Matplotlib, etc.)</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Setting Up VS Code for Python</Title>
        <Text size="md" mb="md">Follow these steps to configure VS Code for Python development:</Text>
        <List type="ordered" spacing="sm" mb="md">
          <List.Item>
            Download and install VS Code from the{" "}
            <Anchor href="https://code.visualstudio.com/" target="_blank" rel="noopener noreferrer">
              official website
            </Anchor>
          </List.Item>
          <List.Item>
            Install the{" "}
            <Anchor href="https://marketplace.visualstudio.com/items?itemName=ms-python.python" target="_blank" rel="noopener noreferrer">
              Python extension by Microsoft
            </Anchor>
          </List.Item>
          <List.Item>Configure the Python interpreter</List.Item>
        </List>
        <CodeBlock code={`# Open VS Code and press Ctrl+Shift+P (Cmd+Shift+P on Mac)
# Type "Python: Select Interpreter"
# Choose your Python installation`} language="python" />
      </div>

      <div data-slide>
        <Title order={2} mb="md">Essential VS Code Extensions for Data Science</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <Text fw={500} span>Python:</Text> Core Python language support (
            <Anchor href="https://marketplace.visualstudio.com/items?itemName=ms-python.python" target="_blank" rel="noopener noreferrer">
              Extension
            </Anchor>
            )
          </List.Item>
          <List.Item>
            <Text fw={500} span>Jupyter:</Text> Jupyter notebook support in VS Code (
            <Anchor href="https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter" target="_blank" rel="noopener noreferrer">
              Extension
            </Anchor>
            )
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Using Terminal Directly in IDE</Title>
        <Text size="md" mb="md">Modern IDEs provide integrated terminals that allow you to run commands without leaving your development environment.</Text>
        <Title order={3} mb="sm">Opening Terminal in VS Code</Title>
        <CodeBlock code={`# Windows/Linux: Ctrl + \`
# Mac: Cmd + \`
# Or use menu: Terminal → New Terminal`} language="bash" />
        <Text size="md" mb="md">Benefits of integrated terminal:</Text>
        <List spacing="sm" mb="md">
          <List.Item>Stay within your development environment</List.Item>
          <List.Item>Easy file navigation relative to your project</List.Item>
          <List.Item>Run Python scripts and install packages directly</List.Item>
          <List.Item>Multiple terminal sessions in tabs</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Configure Python Interpreter in IDE</Title>
        <Text size="md" mb="md">Setting up the correct Python interpreter is crucial for your data science projects.</Text>
        <Title order={3} mb="sm">Selecting Python Interpreter in VS Code</Title>
        <CodeBlock code={`# Method 1: Command Palette
# Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
# Type: "Python: Select Interpreter"
# Choose from available Python installations

# Method 2: Status Bar
# Click on Python version in bottom status bar
# Select interpreter from the list`} language="python" />
      </div>

      <div data-slide>
        <Title order={2} mb="md">Using Git Directly in IDE</Title>
        <Text size="md" mb="md">IDEs provide powerful Git integration for version control without leaving your editor.</Text>
        <Title order={3} mb="sm">Git Features in VS Code</Title>
        <List spacing="sm" mb="md">
          <List.Item><Text fw={500} span>Source Control panel:</Text> View changes, stage files, commit</List.Item>
          <List.Item><Text fw={500} span>Diff view:</Text> Side-by-side comparison of changes</List.Item>
          <List.Item><Text fw={500} span>Branch management:</Text> Switch, create, and merge branches</List.Item>
          <List.Item><Text fw={500} span>Conflict resolution:</Text> Visual merge conflict editor</List.Item>
        </List>
        <CodeBlock code={`# Common Git operations in VS Code:
# 1. Open Source Control: Ctrl+Shift+G
# 2. Stage changes: Click + next to files
# 3. Commit: Type message and press Ctrl+Enter
# 4. Push/Pull: Click ... for more options`} language="bash" />
      </div>

      <div data-slide>
        <Title order={2} mb="md">AI Code Assistant</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <Text fw={500} span>GitHub Copilot:</Text> AI-powered code completion tool that helps accelerate development.{" "}
            <Anchor href="https://github.com/features/copilot" target="_blank" rel="noopener noreferrer">
              (Website)
            </Anchor>
          </List.Item>
          <List.Item>
            <Text fw={500} span>Claude Code:</Text> AI-powered coding assistant by Anthropic, focused on privacy and reliability{" "}
            <Anchor href="https://www.anthropic.com/claude" target="_blank" rel="noopener noreferrer">
              (Website)
            </Anchor>
          </List.Item>
          <List.Item>
            <Text fw={500} span>Cursor:</Text> AI-first code editor with built-in code completion and chat features{" "}
            <Anchor href="https://www.cursor.so/" target="_blank" rel="noopener noreferrer">
              (Website)
            </Anchor>
          </List.Item>
          <List.Item>
            <Text fw={500} span>Amazon CodeWhisperer:</Text> AI coding companion integrated with AWS services{" "}
            <Anchor href="https://aws.amazon.com/codewhisperer/" target="_blank" rel="noopener noreferrer">
              (Website)
            </Anchor>
          </List.Item>
          <List.Item>
            <Text fw={500} span>Tabnine:</Text> AI code completion supporting multiple languages and IDEs{" "}
            <Anchor href="https://www.tabnine.com/" target="_blank" rel="noopener noreferrer">
              (Website)
            </Anchor>
          </List.Item>
        </List>

        <Text size="md" mb="md">Tips for effective AI Code Assistant usage:</Text>
        <List spacing="sm" mb="md">
          <List.Item>Write clear, descriptive comments</List.Item>
          <List.Item>Review suggestions before accepting</List.Item>
          <List.Item>Use Tab to accept, Escape to dismiss</List.Item>
          <List.Item>AI Code Assistant learns from context in your file (no privacy concerns)</List.Item>
        </List>
      </div>
    </Container>
  );
};

export default Ide;