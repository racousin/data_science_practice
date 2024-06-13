#!/bin/bash

# This script runs tests for module2 and expects a username as a parameter
# Usage: ./run_tests.sh <username>

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <username> <aws-access-key-id> <aws-secret-access-key> <aws-region>"
  exit 1
fi

USERNAME=$1
PACKAGE_DIR="${USERNAME}/module2/mysupertools"
TESTS_DIR="../../../tests/module2/mysupertools/tests"  # Adjust this path if necessary

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Navigate to the package directory
cd "$PACKAGE_DIR" || exit 1

echo "Current working directory: $(pwd)"
echo "Listing contents of the current directory:"
ls -la

export PYTHONPATH="$PYTHONPATH:$(pwd)/.."

# Install the package using pip
pip install .

# Ensure pytest is installed
pip install pytest

# Run pytest to execute the tests, specifying the directory where the tests are located
pytest "$TESTS_DIR"

# Check the outcome of the tests
if [ "$?" -eq 0 ]; then
  echo "All tests passed successfully for module2."
  deactivate
else
  echo "Tests failed for module2."
  deactivate
  exit 1
fi
