import React from "react";
import { Container, Grid } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import EvaluationModal from "components/EvaluationModal";
const Exercise2 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 2: Advanced Python Package Creation</h1>
      <p>
        In this exercise, you will extend the `mysupertools` package from Exercise 1 by adding more functionality, 
        implementing unit tests with pytest, and creating a command-line interface (CLI) for your package.
      </p>
      <Grid>
        <Grid.Col>
          <h2>Instructions</h2>
          <ol>
            <li>Your final directory structure should look like this:</li>
            <CodeBlock
              code={`$username/module2/mysupertools2/
                ├── setup.py
                ├── README.md
                └── mysupertools2/
                    ├── __init__.py
                    ├── cli.py
                    ├── math_operations.py
                    └── tests/
                        ├── __init__.py
                        └── test_math_operations.py`}
            />
            <li>
              In <code>math_operations.py</code>, implement the following functions:
              <ul>
                <li><code>multiply(a, b)</code>: Multiply two numbers (from Exercise 1)</li>
                <li><code>fibonacci(n)</code>: Generate the nth Fibonacci number</li>
              </ul>
            </li>
            <li>
      Create a <code>cli.py</code> file that implements a command-line interface for your package using 
      the <code>argparse</code> module. The CLI should allow users to perform the operations implemented in your package.
      <p>Here's how you can structure your CLI:</p>
      <ol type="a">
        <li>Import the necessary modules and functions:</li>
        <CodeBlock
          code={`import argparse
from mysupertools2.math_operations import multiply, fibonacci`}
          language="python"
        />
        <li>Create a main function that sets up the argument parser:</li>
        <CodeBlock
          code={`def main():
    parser = argparse.ArgumentParser(description='mysupertools2 CLI')
    # Add arguments here
    args = parser.parse_args()`}
          language="python"
        />
        <li>Add arguments to your parser:</li>
        <ul>
          <li>Add an argument for the operation (multiply or fibonacci)</li>
          <li>Add an argument for the numbers to operate on</li>
          <li>Hint: Use <code>parser.add_argument()</code> method</li>
        </ul>
        <li>Implement the logic to handle different operations:</li>
        <ul>
          <li>Check which operation was selected (<code>args.operation</code>)</li>
          <li>Validate the number of arguments provided</li>
          <li>Call the appropriate function and print the result</li>
        </ul>
        <li>Don't forget to add the following at the end of your file:</li>
        <CodeBlock
          code={`if __name__ == '__main__':
    main()`}
          language="python"
        />
      </ol>
      <p>Example usage of your CLI should look like this:</p>
      <CodeBlock
        code={`$ mysupertools2 multiply 4 5
4 * 5 = 20
$ mysupertools2 fibonacci 8
The 8th Fibonacci number is 21`}
      />
      <p>Remember to handle errors gracefully, such as when the wrong number of arguments is provided.</p>
    </li>
    <li>
      Write unit tests for all functions in <code>test_math_operations.py</code> using <code>pytest</code>.
      <p>Here's a guide on how to structure your unit tests:</p>
      <ol type="a">
        <li>Import the necessary modules and functions:</li>
        <CodeBlock
          code={`import pytest
from mysupertools2.math_operations import multiply, fibonacci`}
          language="python"
        />
        <li>Write test functions for each operation:</li>
        <ul>
          <li>Test functions should start with "test_"</li>
          <li>Use descriptive names for your test functions</li>
          <li>Example: <code>def test_multiply():</code></li>
        </ul>
        <li>Inside each test function, use assertions to check expected outcomes:</li>
        <ul>
          <li>Use the <code>assert</code> statement to check if the function returns the expected result</li>
          <li>Test multiple scenarios, including edge cases</li>
          <li>Example: <code>assert multiply(2, 3) == 6</code></li>
        </ul>
        <li>For the Fibonacci function, consider these test cases:</li>
        <ul>
          <li>Test with input 0</li>
          <li>Test with input 1</li>
          <li>Test with a larger number (e.g., 6 or 10)</li>
          <li>Test with a negative number (should raise a ValueError)</li>
        </ul>
        <li>To test for raised exceptions, use pytest's <code>raises</code> context manager:</li>
        <CodeBlock
          code={`def test_fibonacci_negative():
    with pytest.raises(ValueError):
        fibonacci(-1)`}
          language="python"
        />
      </ol>
      <p>Remember to test both normal cases and edge cases for each function. Your tests should cover different scenarios to ensure your functions work correctly under various conditions.</p>
      <p>To run your tests, use the following command in your terminal:</p>
      <CodeBlock
        code={`pytest mysupertools2/tests/`}
      />
      <p>Pytest will automatically discover and run all test functions in files that start with "test_" in the specified directory.</p>
    </li>
    <li>
      Update the <code>setup.py</code> file to include the new modules, CLI entry point, and dependencies. 
      Here's a guide on what to include:
      <ol type="a">
        <li>Import the necessary setup tools:</li>
        <CodeBlock
          code={`from setuptools import setup, find_packages`}
          language="python"
        />
        <li>Use the setup function to configure your package. Include these key elements:</li>
        <ul>
          <li>name: The name of your package</li>
          <li>version: The current version of your package</li>
          <li>packages: Use find_packages() to automatically find your package</li>
          <li>install_requires: List your package dependencies</li>
          <li>entry_points: Define your CLI entry point</li>
        </ul>
        <li>Here's a basic structure to follow:</li>
        <CodeBlock
          code={`setup(
    name='mysupertools2',
    version='0.2',
    packages=find_packages(exclude=['mysupertools2.tests']),
    install_requires=['pandas'],
    entry_points={
        'console_scripts': [
            'mysupertools2=mysupertools2.cli:main',
        ],
    },
    # Add other necessary fields here
)`}
          language="python"
        />
        <li>Consider adding these optional fields for more information about your package:</li>
        <ul>
          <li>author: Your name</li>
          <li>description: A short description of your package</li>
          <li>url: The URL for your package's repository</li>
        </ul>
        <p>Remember, the goal is to make your package installable and to set up the CLI entry point. 
        Make sure to include all necessary dependencies and correctly specify the CLI entry point.</p>
      </ol>
      <p>You can also add development dependencies using the 'extras_require' parameter:</p>
<CodeBlock
  code={`extras_require={
    'dev': ['pytest'],
},`}
  language="python"
/>
<p>This allows users to install development dependencies with 'pip install -e .[dev]'</p>
    </li>
            <li>
              Create a <code>README.md</code> file with instructions on how to install and use your package, 
              including examples of using the CLI.
            </li>
          </ol>
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <h2>Testing Your Code</h2>
          <p>
            To ensure your package is working correctly, follow these steps to test your code:
          </p>
          <ol>
            <li>Install your package with development dependencies:</li>
            <CodeBlock code={`pip install -e .$username/module2/mysupertools2[dev]`} />
            <li>Run the unit tests:</li>
            <CodeBlock code={`pytest mysupertools2/tests/`} />
            <li>Test the CLI:</li>
            <CodeBlock code={`mysupertools2 multiply 4 7
mysupertools2 fibonacci 10`} />
            <li>Import and use your package in a Python session:</li>
            <CodeBlock
              code={`from mysupertools2.math_operations import multiply, fibonacci
result = multiply(10, 5)
fib_number = fibonacci(8)
print(f"10 * 5 = {result}")
print(f"8th Fibonacci number: {fib_number}")
`}
              language="python"
            />
          </ol>
        </Grid.Col>
      </Grid>
    </Container>
  );
};
export default Exercise2;