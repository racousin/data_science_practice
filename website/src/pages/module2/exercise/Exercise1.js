import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Exercise1 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 1: Creating a Python Package</h1>
      <p>
        In this exercise, you will create a Python package named `mysupertools`
        with a module that contains a function to multiply two values. This
        function will return the product if both arguments are numbers, and the
        string "error" otherwise.
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>
              Create a new directory called <code>mysupertools</code>:
            </li>
            <CodeBlock code={`mkdir mysupertools`} />
            <li>
              Navigate to the <code>mysupertools</code> directory:
            </li>
            <CodeBlock code={`cd mysupertools`} />
            <li>
              Create a new directory inside <code>mysupertools</code> called{" "}
              <code>tool</code>:
            </li>
            <CodeBlock code={`mkdir tool`} />
            <li>
              Create a Python file inside the <code>tool</code> directory called{" "}
              <code>multiplication_a_b.py</code>:
            </li>
            <CodeBlock code={`touch tool/multiplication_a_b.py`} />
            <li>
              Edit the <code>multiplication_a_b.py</code> file to add the
              following function:
            </li>
            <CodeBlock
              code={`def multiply(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    else:
        return "error"
`}
            />
            <li>
              Create a <code>setup.py</code> file in the{" "}
              <code>mysupertools</code> directory with the following content to
              make it a package:
            </li>
            <CodeBlock
              code={`from setuptools import setup, find_packages

setup(
    name='mysupertools',
    version='0.1',
    packages=find_packages(),
)
`}
            />
            <li>
              Create a <code>__init__.py</code> file inside both the{" "}
              <code>mysupertools</code> and <code>tool</code> directories to
              make them Python packages:
            </li>
            <CodeBlock code={`touch __init__.py`} />
            <CodeBlock code={`touch tool/__init__.py`} />
            <li>Your final directory structure should look like this:</li>
            <CodeBlock
              code={`mysupertools/
├── __init__.py
├── setup.py
└── tool/
    ├── __init__.py
    └── multiplication_a_b.py`}
            />
            <li>Commit and push your changes to your GitHub repository.</li>
          </ol>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Testing Your Code</h2>
          <p>
            To test your code, you can use the following test script that will
            be run automatically by the CI/CD system to validate your solution:
          </p>
          <CodeBlock
            code={`# mysupertools/tests/test_multiplication.py

from mysupertools.tool.multiplication_a_b import multiply

def test_multiply_numbers():
    assert multiply(4, 5) == 20
    assert multiply(-1, 5) == -5

def test_multiply_errors():
    assert multiply("a", 5) == "error"
    assert multiply(None, 5) == "error"
`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise1;
