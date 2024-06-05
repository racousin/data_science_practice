import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Exercise2 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 2: Setting Up a Python Package</h1>
      <p>
        In this exercise, you will structure your project as a simple Python
        package that can be installed using pip.
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>Navigate to the `mysupertools` directory:</li>
            <CodeBlock code={`cd mysupertools`} />
            <li>Create a new file called `setup.py`:</li>
            <CodeBlock code={`touch setup.py`} />
            <li>
              Open `setup.py` in a text editor and add the following code:
            </li>
            <CodeBlock
              code={`from setuptools import setup

setup(
    name='mysupertools',
    version='1.0',
    description='A simple Python package',
    author='Your Name',
    author_email='youremail@example.com',
    url='https://github.com/yourusername/mysupertools',
    packages=['mysupertools.tool'],
)`}
            />
            <li>Create a new file called `README.md`:</li>
            <CodeBlock code={`touch README.md`} />
            <li>
              Open `README.md` in a text editor and add the following content:
            </li>
            <CodeBlock
              code={`# MySuperTools

A simple Python package that provides a multiplication function.

## Installation

To install the package, run the following command:

\`\`\`
pip install .
\`\`\`

## Usage

To use the multiplication function, import the \`tool\` module and call the \`multiply\` function:

\`\`\`python
from mysupertools.tool import multiplication_a_b

result = multiplication_a_b.multiply(2, 3)
print(result)  # Output: 6
\`\`\`
`}
            />
          </ol>
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise2;
