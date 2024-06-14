import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const BestPracticesAndResources = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Best Practices and Resources</h1>
      <p>
        This section covers best practices in Python development and provides
        resources for further learning and improvement. Following these
        practices will help you write clean, efficient, and maintainable code.
      </p>

      <Row>
        <Col>
          <h3 id="other-package-managers">Other Package Managers</h3>
          <p>
            While <code>pip</code> is the standard package manager for Python,
            there are other tools available that provide additional features and
            benefits.
          </p>

          <h4>Poetry</h4>
          <p>
            Poetry is a tool for dependency management and packaging in Python.
            It allows you to declare the libraries your project depends on and
            it will manage (install/update) them for you.
          </p>
          <CodeBlock code={`pip install poetry`} />
          <p>Initialize a new project with Poetry:</p>
          <CodeBlock code={`poetry init`} />
          <p>Add a dependency:</p>
          <CodeBlock code={`poetry add requests`} />

          <h4>Conda</h4>
          <p>
            Conda is an open-source package management and environment
            management system that runs on Windows, macOS, and Linux. It quickly
            installs, runs, and updates packages and their dependencies.
          </p>
          <CodeBlock code={`conda create --name myenv`} />
          <p>Activate the new environment:</p>
          <CodeBlock code={`conda activate myenv`} />
          <p>Install a package:</p>
          <CodeBlock code={`conda install numpy`} />
        </Col>
      </Row>

      <Row>
        <Col>
          <h3 id="testing-and-unit-tests">Testing and Unit Tests</h3>
          <p>
            Writing tests for your code is a crucial practice to ensure
            functionality and to catch bugs early. Python has several testing
            frameworks that make it easy to write and run tests.
          </p>

          <h4>pytest</h4>
          <p>
            <code>pytest</code> is a popular testing framework for Python that
            makes it easy to write simple and scalable test cases.
          </p>
          <CodeBlock code={`pip install pytest`} />
          <p>
            Create a test file (e.g., <code>test_example.py</code>):
          </p>
          <CodeBlock
            code={`def test_addition():
    assert 1 + 1 == 2`}
            language={"python"}
          />
          <p>Run the tests:</p>
          <CodeBlock code={`pytest`} />

          <h4>Unittest</h4>
          <p>
            <code>unittest</code> is the built-in testing framework in Python.
            It provides a test discovery mechanism and a base class for defining
            tests.
          </p>
          <CodeBlock
            code={`import unittest

class TestMath(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    unittest.main()`}
            language={"python"}
          />
        </Col>
      </Row>

      <Row>
        <Col>
          <h3 id="syntax-and-linting">Syntax and Linting</h3>
          <p>
            Maintaining a consistent code style is important for readability and
            maintainability. There are several tools available to help enforce a
            consistent style.
          </p>

          <h4>Black</h4>
          <p>
            Black is an uncompromising code formatter for Python. It formats
            your code to comply with PEP 8 standards.
          </p>
          <CodeBlock code={`pip install black`} />
          <p>Format your code:</p>
          <CodeBlock code={`black myscript.py`} />

          <h4>Flake8</h4>
          <p>
            Flake8 is a tool for checking the style and quality of Python code.
            It combines PEP 8 compliance checks with linting and complexity
            checking.
          </p>
          <CodeBlock code={`pip install flake8`} />
          <p>Run Flake8:</p>
          <CodeBlock code={`flake8 myscript.py`} />
        </Col>
      </Row>

      <Row>
        <Col>
          <h3 id="resources">Additional Resources</h3>
          <p>
            Here are some additional resources to help you further your Python
            knowledge and improve your coding skills.
          </p>

          <h4>Books</h4>
          <ul>
            <li>
              <a
                href="https://automatetheboringstuff.com/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Automate the Boring Stuff with Python
              </a>{" "}
              by Al Sweigart
            </li>
          </ul>

          <h4>Performance Optimization</h4>
          <p>
            Improving the performance of your Python code can be crucial in
            resource-intensive applications. Here are some tools and techniques:
          </p>
          <ul>
            <li>
              <a
                href="https://docs.python.org/3/library/profile.html"
                target="_blank"
                rel="noopener noreferrer"
              >
                cProfile
              </a>{" "}
              - A built-in Python module for profiling your code.
            </li>
            <li>
              <a
                href="https://pandas.pydata.org/pandas-docs/stable/user_guide/scale.html"
                target="_blank"
                rel="noopener noreferrer"
              >
                Pandas Performance
              </a>{" "}
              - Best practices for optimizing Pandas operations.
            </li>
          </ul>
        </Col>
      </Row>
    </Container>
  );
};

export default BestPracticesAndResources;
