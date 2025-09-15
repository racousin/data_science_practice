import React from "react";
import { Container, Grid, Title, Text, List, Anchor, Code } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
const InstallPython = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">Installing Python</Title>
      <Grid>
        <Grid.Col span={{ md: 12 }}>
          <Title order={3} id="windows" mb="sm">Windows</Title>
          <Text size="md" mb="sm">Follow these steps to install Python on Windows:</Text>
          <List type="ordered" spacing="sm">
            <List.Item>
              Download the latest version of Python from the official Python
              website:
              <br />
              <Anchor
                href="https://www.python.org/downloads/windows/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Python Downloads for Windows
              </Anchor>
            </List.Item>
            <List.Item>
              Run the downloaded installer. Ensure to select "Add Python 3.x to
              PATH" at the start of the installation process.
            </List.Item>
            <List.Item>Follow the installation prompts to complete the setup.</List.Item>
          </List>
          <Title order={3} id="mac" mb="sm">MacOS</Title>
          <Text size="md" mb="sm">
            MacOS comes with Python pre-installed. To check the installed
            version:
          </Text>
          <List type="ordered" spacing="sm">
            <List.Item>
              Open a terminal window and type the following command to check
              your current Python version:
              <CodeBlock code={`python3 --version`} />
              <CodeBlock
                showCopy={false}
                code={`$ python3 --version
Python 3.10.12
`}
                language=""
              />
              or try:
              <CodeBlock code={`python --version`} />
            </List.Item>
            <List.Item>
              If you need a newer version, consider installing Python via{" "}
              <Anchor
                href="https://brew.sh/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Homebrew
              </Anchor>{" "}
              or directly from the Python website.
            </List.Item>
          </List>
          <Title order={3} id="linux" mb="sm">Linux</Title>
          <Text size="md" mb="sm">
            Most Linux distributions come with Python pre-installed. To verify
            or install Python, you can use your distribution's package manager:
          </Text>
          <List type="ordered" spacing="sm">
            <List.Item>
              Open a terminal window and check the installed version of Python:
              <CodeBlock code={`python3 --version`} />
              <CodeBlock
                showCopy={false}
                code={`$ python3 --version
Python 3.10.12
`}
                language=""
              />
              or try:
              <CodeBlock code={`python --version`} />
            </List.Item>
            <List.Item>
              If Python is not installed, or if you need a different version,
              use your distribution's package manager to install Python. For
              example, on Ubuntu, you would use:
              <CodeBlock code={`sudo apt-get install python3`} />
            </List.Item>
          </List>
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col span={{ md: 12 }}>
          <Title order={3} id="install-pip" mb="sm">Install pip</Title>
          <Text size="md" mb="sm">
            <Code>pip</Code> is Python's package installer and is included by
            default with Python versions 3.4 and above. It's crucial for
            managing third-party Python packages. Here's how to ensure it is
            installed and up to date:
          </Text>
          <List type="ordered" spacing="sm">
            <List.Item>
              To check if <Code>pip</Code> is installed, open a terminal or
              command prompt and type:
              <CodeBlock code={`pip --version`} />
              <CodeBlock
                showCopy={false}
                code={`$ pip --version
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
`}
                language=""
              />
            </List.Item>
            <List.Item>
              If <Code>pip</Code> is not installed, you can install it by
              downloading <Code>get-pip.py</Code>:
              <br />
              <Anchor
                href="https://bootstrap.pypa.io/get-pip.py"
                target="_blank"
                rel="noopener noreferrer"
              >
                Download get-pip.py
              </Anchor>
            </List.Item>
            <List.Item>
              After downloading, run the following command in the directory
              where <Code>get-pip.py</Code> is located:
              <CodeBlock code={`python get-pip.py`} />
            </List.Item>
            <List.Item>
              To upgrade an existing <Code>pip</Code> installation to the latest
              version, use:
              <CodeBlock code={`pip install --upgrade pip`} />
            </List.Item>
          </List>
          <Text size="md" mt="sm">
            Ensuring <Code>pip</Code> is installed and up to date allows you to
            easily manage and install packages, which are often necessary for
            development projects.
          </Text>
        </Grid.Col>
      </Grid>
    </Container>
  );
};
export default InstallPython;
