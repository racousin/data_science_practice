import React from "react";
import { Container, Grid } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const UnitAndIntegrationTests = () => {
  return (
    <Container fluid>
      <div data-slide>
        <h2>Testing in Python</h2>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <h4>Why Testing Matters</h4>
            <p>
              Testing ensures your code works as expected, helps catch bugs early,
              and provides confidence when making changes. It's essential for
              maintaining reliable software systems.
            </p>
            <h4>Types of Tests</h4>
            <ul>
              <li><strong>Unit Tests:</strong> Test individual functions or methods in isolation</li>
              <li><strong>Integration Tests:</strong> Test how multiple components work together</li>
            </ul>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h3>Testing with pytest</h3>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <p>
              pytest is a popular third-party testing framework that makes writing
              tests more straightforward with less boilerplate code.
            </p>
            <h4>Installation</h4>
            <CodeBlock
              code={`pip install pytest`}
              language="bash"
            />

            <h4>Simple pytest Example</h4>
            <CodeBlock
              code={`def add_numbers(a, b):
    return a + b

def test_add_numbers():
    assert add_numbers(2, 3) == 5`}
              language="python"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h3>Running Tests</h3>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <h4>Running pytest</h4>
            <CodeBlock
              code={`pytest test_module.py`}
              language="bash"
            />

            <h4>Running All Tests</h4>
            <CodeBlock
              code={`pytest`}
              language="bash"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h3>Integration Testing</h3>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <p>
              Integration tests verify that different parts of your system work
              correctly together. They test the interaction between modules,
              databases, APIs, and external services.
            </p>
            <h4>Example Integration Test</h4>
            <CodeBlock
              code={`def test_user_registration_workflow():
    # Test complete user registration process
    user_data = {"name": "John", "email": "john@example.com"}
    user = create_user(user_data)
    saved_user = get_user_by_email("john@example.com")
    assert saved_user.name == "John"`}
              language="python"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <h3>Test Organization and Best Practices</h3>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <h4>Test Directory Structure</h4>
            <CodeBlock
              code={`project/
├── src/
│   └── mymodule.py
└── tests/
    ├── test_mymodule.py
    └── integration/
        └── test_workflows.py`}
              language="text"
            />

            <h4>Best Practices</h4>
            <ul>
              <li>Write descriptive test names that explain what is being tested</li>
              <li>Keep tests independent and isolated</li>
              <li>Use fixtures for test data setup</li>
              <li>Aim for good test coverage but focus on critical paths</li>
            </ul>
          </Grid.Col>
        </Grid>
      </div>
    </Container>
  );
};

export default UnitAndIntegrationTests;