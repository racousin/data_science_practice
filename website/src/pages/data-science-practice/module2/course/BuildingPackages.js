import React from "react";
import { Container, Grid, Title, Text, List, Anchor, Alert } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";
const BuildingPackages = () => {
  return (
    <Container fluid>
      <Title order={1} mb="xl">Building Python Packages</Title>
      <Grid>
        <Grid.Col>
          <Title order={3} id="package-structure" mb="md">Package Project Structure</Title>
          <Text mb="md">Here's what the directory structure should look like:</Text>
          <CodeBlock
            code={`mypackage/
├── mypackage/
│   ├── __init__.py
│   ├── functions1.py
│   ├── functions2.py
│   ├── tools/
│       ├── __init__.py
│       └── tools1.py
├── pyproject.toml
└── README.md`}
            language="text"
          />
          <Text mb="md">Here's a brief explanation of each component:</Text>
          <List spacing="sm">
            <List.Item>
              <strong>mypackage/:</strong> The root directory of your package project.
            </List.Item>
            <List.Item>
              <strong>mypackage/mypackage/:</strong> The directory containing your package's modules.
            </List.Item>
            <List.Item>
              <strong>__init__.py:</strong> A file that initializes the package and can contain package-level variables and imports.
            </List.Item>
            <List.Item>
              <strong>functions1.py:</strong> A module within the package. You can add more modules as needed.
            </List.Item>
            <List.Item>
              <strong>functions2.py:</strong> Another module within the package.
            </List.Item>
            <List.Item>
              <strong>pyproject.toml:</strong> The modern configuration file containing metadata about the package and build instructions.
            </List.Item>
            <List.Item>
              <strong>README.md:</strong> A README file providing an overview of the package, installation instructions, and usage examples.
            </List.Item>
          </List>
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <Title order={3} id="create-directory" mb="md">Create a New Directory for Your Package</Title>
          <Text mb="md">First, create a new directory where your package will reside:</Text>
          <CodeBlock code={`mkdir mypackage`} language="bash" />
          <Text mb="md">Navigate into the newly created directory:</Text>
          <CodeBlock code={`cd mypackage`} language="bash" />
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <Title order={3} id="pyproject-file" mb="md">Create the `pyproject.toml` File</Title>
          <Alert icon={<IconInfoCircle />} color="blue" mb="md">
            Modern Python packaging uses `pyproject.toml` instead of `setup.py`. While `setup.py` is still supported, it's considered legacy and `pyproject.toml` is the recommended approach since PEP 518.
          </Alert>
          <Text mb="md">
            The `pyproject.toml` file is the modern standard for Python project configuration. It can be created manually or generated using tools like Poetry, Hatch, or PDM.
          </Text>

          <Text mb="sm"><strong>Essential sections in pyproject.toml:</strong></Text>
          <List spacing="xs" mb="md">
            <List.Item><strong>[build-system]</strong> - Required: Specifies how to build your package</List.Item>
            <List.Item><strong>[project]</strong> - Required: Contains package metadata (name, version, description)</List.Item>
            <List.Item><strong>[project.urls]</strong> - Optional but recommended: Project links</List.Item>
            <List.Item><strong>[project.dependencies]</strong> - Optional: Runtime dependencies</List.Item>
          </List>

          <Text mb="md">Here's a minimal example you can create manually:</Text>
          <CodeBlock
            code={`[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "1.0.0"
description = "A simple Python package"
readme = "README.md"
authors = [
    {name = "Your Name", email = "youremail@example.com"}
]
requires-python = ">=3.8"

[project.urls]
"Homepage" = "https://github.com/yourusername/mypackage"
"Bug Reports" = "https://github.com/yourusername/mypackage/issues"
"Source" = "https://github.com/yourusername/mypackage"`}
            language="toml"
          />

          <Text mb="md"><strong>Alternative: Using Poetry to generate pyproject.toml</strong></Text>
          <Text mb="sm">Poetry can automatically create and manage your pyproject.toml:</Text>
          <CodeBlock
            code={`# Install Poetry first
pip install poetry

# Create a new project with pyproject.toml
poetry new mypackage

# Or initialize in existing directory
poetry init`}
            language="bash"
          />

        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <Title order={3} id="package-directory" mb="md">Create the Package Directory</Title>
          <Text mb="md">Create a new directory with the same name as your package:</Text>
          <CodeBlock code={`mkdir mypackage`} language="bash" />
          <Text mb="md">
            Inside this directory, create an `__init__.py` file to indicate that this directory should be treated as a package:
          </Text>
          <CodeBlock code={`touch mypackage/__init__.py`} language="bash" />
          <Text mb="md">
            The `__init__.py` file can be empty, or you can put initialization code for your package there. Here are some common usage examples:
          </Text>
          <Text mb="sm"><strong>Empty __init__.py (minimal approach):</strong></Text>
          <CodeBlock
            code={`# This file can be empty
# It just marks the directory as a Python package`}
            language="python"
          />
          <Text mb="sm"><strong>__init__.py with version information:</strong></Text>
          <CodeBlock
            code={`__version__ = "1.0.0"
__author__ = "Your Name"`}
            language="python"
          />
          <Text mb="sm"><strong>__init__.py with convenient imports:</strong></Text>
          <CodeBlock
            code={`from .functions1 import hello
from .functions2 import goodbye
from .tools.tools1 import utility_function

# Now users can import directly: from mypackage import hello
__all__ = ["hello", "goodbye", "utility_function"]`}
            language="python"
          />
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <Title order={3} id="add-modules" mb="md">Add Modules to Your Package</Title>
          <Text mb="md">
            Create Python modules within the `mypackage` directory. For example:
          </Text>
          <CodeBlock code={`touch mypackage/functions1.py mypackage/functions2.py`} language="bash" />
          <Text mb="md">
            Add your code to these modules. For example, `functions1.py` might contain:
          </Text>
          <CodeBlock
            code={`def hello():
    print("Hello from functions1!")`}
            language="python"
          />
          <Text mb="md">And `functions2.py` might contain:</Text>
          <CodeBlock
            code={`def goodbye():
    print("Goodbye from functions2!")`}
            language="python"
          />
        </Grid.Col>
      </Grid>
      <Grid>
  <Grid.Col>
    <Title order={3} id="install-your-pkg" mb="md">Install Your Package</Title>
    <Text mb="md">
      To install your package locally, use the following command:
    </Text>
    <CodeBlock code={`pip install mypackage/`} language="bash" />
    <Text mb="md">
      This will install the package directly from the local directory. However, during development, it is often useful to install the package in "editable" mode. This allows you to make changes to the code without needing to reinstall it every time. To install the package in editable mode, run:
    </Text>
    <CodeBlock code={`pip install -e mypackage/`} language="bash" />
    <Text mb="md">
      The <code>-e</code> flag stands for "editable", meaning the package is linked to your current working directory, so any changes made to the package will immediately be reflected without needing to reinstall.
    </Text>
    <Text mb="md">
            Now you can use your package in Python just like any other package:
          </Text>
          <CodeBlock
            code={`import mypackage.functions1 as fct1
import mypackage.functions2 as fct2
fct1.hello()
fct2.goodbye()`}
            language="python"
          />
  </Grid.Col>
</Grid>
      <Grid>
        <Grid.Col>
          <Title order={3} id="build-package" mb="md">Build the Package</Title>
          <Text mb="md">Use the following command to build your package:</Text>
          <CodeBlock code={`python -m build`} language="bash" />
          <Text mb="md">
            This command creates both source and wheel distributions containing your package. The distribution files are stored in the `dist` directory. You may need to install the build tool first:
          </Text>
          <CodeBlock code={`pip install build`} language="bash" />
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <Title order={3} id="install-package-built" mb="md">Install the built Package</Title>
          <Text mb="md">Install the newly created package using pip:</Text>
          <CodeBlock code={`pip install dist/mypackage-1.0.0-py3-none-any.whl`} language="bash" />
          <Text mb="md">
            Replace the filename with the actual name of the generated wheel file in your `dist` directory.
          </Text>
          <Text mb="md">
            Now you can use your package in Python just like any other package:
          </Text>
          <CodeBlock
            code={`import mypackage.functions1 as fct1
import mypackage.functions2 as fct2
fct1.hello()
fct2.goodbye()`}
            language="python"
          />
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <Title order={3} id="publishing-package" mb="md">Publishing Your Package</Title>
          <Text mb="md">
            Once your package is ready, you might want to share it with the world by publishing it to the Python Package Index (PyPI). Here are the basic steps:
          </Text>
          <List type="ordered" spacing="sm">
            <List.Item>Install Twine, a tool for publishing Python packages:</List.Item>
          </List>
          <CodeBlock code={`pip install twine`} language="bash" />
          <List type="ordered" start={2} spacing="sm">
            <List.Item>Upload your package to PyPI:</List.Item>
          </List>
          <CodeBlock code={`twine upload dist/*`} language="bash" />
          <Text mb="md">
            You'll need a PyPI account and will be prompted to enter your credentials.
          </Text>
          <Text mb="md">
            For detailed instructions on how to publish your package, visit the{" "}
            <Anchor
              href="https://packaging.python.org/tutorials/packaging-projects/"
              target="_blank"
              rel="noopener noreferrer"
            >
              official Python packaging guide
            </Anchor>
            .
          </Text>
        </Grid.Col>
      </Grid>
    </Container>
  );
};
export default BuildingPackages;
