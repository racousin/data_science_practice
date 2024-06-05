import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const BuildingPackages = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Building Packages</h1>
      <p>
        In this section, you will learn how to build and distribute your own
        Python packages using setuptools.
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>Create a new directory for your package:</li>
            <CodeBlock code={`mkdir mypackage`} />
            <li>Navigate to the package directory:</li>
            <CodeBlock code={`cd mypackage`} />
            <li>Create a new file called `setup.py`:</li>
            <CodeBlock code={`touch setup.py`} />
            <li>
              Open `setup.py` in a text editor and add the following code:
            </li>
            <CodeBlock
              code={`from setuptools import setup

setup(
    name='mypackage',
    version='1.0',
    description='A simple Python package',
    author='Your Name',
    author_email='youremail@example.com',
    url='https://github.com/yourusername/mypackage',
    packages=['mypackage'],
)`}
            />
            <li>Create a new directory called `mypackage`:</li>
            <CodeBlock code={`mkdir mypackage`} />
            <li>
              Create a new file called `__init__.py` inside the `mypackage`
              directory:
            </li>
            <CodeBlock code={`touch mypackage/__init__.py`} />
            <li>Build the package:</li>
            <CodeBlock code={`python setup.py sdist`} />
            <li>Install the package:</li>
            <CodeBlock code={`pip install dist/mypackage-1.0.tar.gz`} />
          </ol>
        </Col>
      </Row>
    </Container>
  );
};

export default BuildingPackages;
