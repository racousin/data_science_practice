import React from "react";
import { Container, Grid } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import EvaluationModal from "components/EvaluationModal";

const Exercise3 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 3: Comprehensive Unit Testing</h1>
      <p>
        In this exercise, you will create a comprehensive testing suite for a data processing package.
        You'll implement both unittest and pytest testing approaches, write test cases for edge cases,
        and organize tests following best practices.
      </p>

      <Grid>
        <Grid.Col>
          <h2>Instructions</h2>
          <ol>
            <li>Your final directory structure should look like this:</li>
            <CodeBlock
              code={`$username/module2/dataprocessor/
                ├── setup.py
                ├── README.md
                └── dataprocessor/
                    ├── __init__.py
                    ├── math_utils.py
                    ├── string_utils.py
                    ├── data_validator.py
                    └── tests/
                        ├── __init__.py
                        ├── test_math_utils_unittest.py
                        ├── test_string_utils_pytest.py
                        └── test_data_validator_pytest.py`}
            />

            <li>
              In <code>math_utils.py</code>, implement these functions:
              <ul>
                <li><code>calculate_mean(numbers)</code>: Calculate arithmetic mean</li>
                <li><code>calculate_median(numbers)</code>: Calculate median value</li>
                <li><code>factorial(n)</code>: Calculate factorial of n</li>
              </ul>
              <CodeBlock
                code={`def calculate_mean(numbers):
    """Calculate the arithmetic mean of a list of numbers."""
    # Implement function that returns mean or raises ValueError for empty list
    pass

def calculate_median(numbers):
    """Calculate the median of a list of numbers."""
    # Implement function that returns median or raises ValueError for empty list
    pass

def factorial(n):
    """Calculate factorial of n."""
    # Implement function that returns factorial or raises ValueError for negative numbers
    pass`}
                language="python"
              />
            </li>

            <li>
              In <code>string_utils.py</code>, implement these functions:
              <ul>
                <li><code>capitalize_words(text)</code>: Capitalize first letter of each word</li>
                <li><code>count_vowels(text)</code>: Count vowels in text</li>
                <li><code>reverse_string(text)</code>: Reverse a string</li>
              </ul>
              <CodeBlock
                code={`def capitalize_words(text):
    """Capitalize the first letter of each word in the text."""
    # Implement function that handles None input appropriately
    pass

def count_vowels(text):
    """Count the number of vowels in the text (case insensitive)."""
    # Implement function that counts a, e, i, o, u
    pass

def reverse_string(text):
    """Reverse the given string."""
    # Implement function that handles None and empty string
    pass`}
                language="python"
              />
            </li>

            <li>
              In <code>data_validator.py</code>, implement these functions:
              <ul>
                <li><code>validate_email(email)</code>: Check if email format is valid</li>
                <li><code>validate_phone(phone)</code>: Check if phone number format is valid</li>
                <li><code>validate_age(age)</code>: Check if age is valid (0-150)</li>
              </ul>
              <CodeBlock
                code={`import re

def validate_email(email):
    """Validate email format using regex."""
    # Implement basic email validation
    pass

def validate_phone(phone):
    """Validate phone number format (xxx-xxx-xxxx)."""
    # Implement phone validation for format xxx-xxx-xxxx
    pass

def validate_age(age):
    """Validate age is between 0 and 150."""
    # Implement age validation
    pass`}
                language="python"
              />
            </li>

            <li>
              Create <code>test_math_utils_unittest.py</code> using Python's unittest framework:
              <CodeBlock
                code={`import unittest
from dataprocessor.math_utils import calculate_mean, calculate_median, factorial

class TestMathUtils(unittest.TestCase):

    def test_calculate_mean_normal(self):
        """Test mean calculation with normal inputs."""
        # Test with positive numbers
        # Test with negative numbers
        # Test with mixed numbers
        pass

    def test_calculate_mean_edge_cases(self):
        """Test mean calculation edge cases."""
        # Test with empty list (should raise ValueError)
        # Test with single number
        # Test with zeros
        pass

    def test_calculate_median_odd_count(self):
        """Test median with odd number of elements."""
        pass

    def test_calculate_median_even_count(self):
        """Test median with even number of elements."""
        pass

    def test_factorial_positive(self):
        """Test factorial with positive integers."""
        # Test factorial(0) == 1
        # Test factorial(5) == 120
        pass

    def test_factorial_negative(self):
        """Test factorial with negative numbers."""
        # Should raise ValueError
        pass

if __name__ == '__main__':
    unittest.main()`}
                language="python"
              />
            </li>

            <li>
              Create <code>test_string_utils_pytest.py</code> using pytest framework:
              <CodeBlock
                code={`import pytest
from dataprocessor.string_utils import capitalize_words, count_vowels, reverse_string

def test_capitalize_words_normal():
    """Test word capitalization with normal inputs."""
    # Test "hello world" -> "Hello World"
    # Test "python programming" -> "Python Programming"
    pass

def test_capitalize_words_edge_cases():
    """Test word capitalization edge cases."""
    # Test empty string
    # Test None input
    # Test single character
    # Test multiple spaces
    pass

def test_count_vowels_normal():
    """Test vowel counting with normal inputs."""
    # Test "hello" -> 2
    # Test "programming" -> 3
    pass

def test_count_vowels_case_insensitive():
    """Test vowel counting is case insensitive."""
    # Test "HELLO" -> 2
    # Test "AeIoU" -> 5
    pass

def test_count_vowels_edge_cases():
    """Test vowel counting edge cases."""
    # Test empty string -> 0
    # Test consonants only -> 0
    # Test None input
    pass

def test_reverse_string_normal():
    """Test string reversal with normal inputs."""
    # Test "hello" -> "olleh"
    # Test "Python" -> "nohtyP"
    pass

def test_reverse_string_edge_cases():
    """Test string reversal edge cases."""
    # Test empty string
    # Test single character
    # Test None input
    pass`}
                language="python"
              />
            </li>

            <li>
              Create <code>test_data_validator_pytest.py</code> with comprehensive validation tests:
              <CodeBlock
                code={`import pytest
from dataprocessor.data_validator import validate_email, validate_phone, validate_age

class TestEmailValidation:

    def test_valid_emails(self):
        """Test valid email formats."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org"
        ]
        # Test each valid email returns True
        pass

    def test_invalid_emails(self):
        """Test invalid email formats."""
        invalid_emails = [
            "notanemail",
            "@example.com",
            "test@",
            "test..double@example.com"
        ]
        # Test each invalid email returns False
        pass

class TestPhoneValidation:

    def test_valid_phones(self):
        """Test valid phone formats."""
        # Test "123-456-7890" -> True
        # Test "999-888-7777" -> True
        pass

    def test_invalid_phones(self):
        """Test invalid phone formats."""
        # Test "1234567890" -> False (no dashes)
        # Test "123-45-6789" -> False (wrong format)
        # Test "abc-def-ghij" -> False (not numbers)
        pass

class TestAgeValidation:

    def test_valid_ages(self):
        """Test valid age ranges."""
        # Test ages 0, 25, 100, 150
        pass

    def test_invalid_ages(self):
        """Test invalid age ranges."""
        # Test negative ages, ages > 150
        pass`}
                language="python"
              />
            </li>

            <li>
              Update your <code>setup.py</code> to include testing dependencies:
              <CodeBlock
                code={`from setuptools import setup, find_packages

setup(
    name='dataprocessor',
    version='1.0.0',
    packages=find_packages(exclude=['dataprocessor.tests']),
    install_requires=[
        'pytest',
    ],
    extras_require={
        'dev': ['pytest', 'pytest-cov'],
    },
    author='Your Name',
    description='A comprehensive data processing package with full test coverage',
    python_requires='>=3.6',
)`}
                language="python"
              />
            </li>
          </ol>
        </Grid.Col>
      </Grid>

      <Grid>
        <Grid.Col>
          <h2>Testing Requirements</h2>
          <p>Your tests must cover the following scenarios:</p>
          <ol>
            <li><strong>Normal Cases:</strong> Test functions with typical valid inputs</li>
            <li><strong>Edge Cases:</strong> Test with empty inputs, None values, boundary conditions</li>
            <li><strong>Error Cases:</strong> Test that appropriate exceptions are raised</li>
            <li><strong>Data Types:</strong> Test with different data types where applicable</li>
          </ol>

          <h3>Specific Test Cases to Implement</h3>
          <ul>
            <li><strong>Math Utils:</strong>
              <ul>
                <li>Mean of [1, 2, 3, 4, 5] should be 3.0</li>
                <li>Median of [1, 2, 3] should be 2</li>
                <li>Median of [1, 2, 3, 4] should be 2.5</li>
                <li>Factorial of 5 should be 120</li>
                <li>Factorial of negative number should raise ValueError</li>
              </ul>
            </li>
            <li><strong>String Utils:</strong>
              <ul>
                <li>capitalize_words("hello world") should return "Hello World"</li>
                <li>count_vowels("programming") should return 3</li>
                <li>reverse_string("python") should return "nohtyp"</li>
              </ul>
            </li>
            <li><strong>Data Validator:</strong>
              <ul>
                <li>validate_email("test@example.com") should return True</li>
                <li>validate_phone("123-456-7890") should return True</li>
                <li>validate_age(25) should return True</li>
                <li>validate_age(-5) should return False</li>
              </ul>
            </li>
          </ul>
        </Grid.Col>
      </Grid>

      <Grid>
        <Grid.Col>
          <h2>Running Your Tests</h2>
          <p>To verify your implementation:</p>
          <ol>
            <li>Install your package with development dependencies:</li>
            <CodeBlock code={`pip install -e $username/module2/dataprocessor[dev]`} />

            <li>Run unittest tests:</li>
            <CodeBlock code={`python -m unittest dataprocessor.tests.test_math_utils_unittest`} />

            <li>Run pytest tests:</li>
            <CodeBlock code={`pytest dataprocessor/tests/test_string_utils_pytest.py -v`} />

            <li>Run all tests:</li>
            <CodeBlock code={`pytest dataprocessor/tests/ -v`} />

            <li>Run tests with coverage report:</li>
            <CodeBlock code={`pytest dataprocessor/tests/ --cov=dataprocessor --cov-report=html`} />

            <li>Test specific functionality in Python:</li>
            <CodeBlock
              code={`from dataprocessor.math_utils import calculate_mean
from dataprocessor.string_utils import capitalize_words
from dataprocessor.data_validator import validate_email

# Test the functions
print(calculate_mean([1, 2, 3, 4, 5]))  # Should print 3.0
print(capitalize_words("hello world"))   # Should print "Hello World"
print(validate_email("test@example.com"))  # Should print True`}
              language="python"
            />
          </ol>
        </Grid.Col>
      </Grid>

      <Grid>
        <Grid.Col>
          <EvaluationModal module={2} />
        </Grid.Col>
      </Grid>
    </Container>
  );
};

export default Exercise3;