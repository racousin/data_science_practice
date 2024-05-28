#!/bin/bash

# Usage: ./run_tests.sh <username> <tp-number>
# Example: ./run_tests.sh alice tp1

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <username> <tp-number>"
  echo "Example: $0 alice tp1"
  exit 1
fi

USERNAME=$1
TP_NUMBER=$2

# Path to the tp-specific test script
TP_TEST_SCRIPT="./tests/$TP_NUMBER/run_tests.sh"

# Check if the test script exists for the given tp
if [ -f "$TP_TEST_SCRIPT" ]; then
  echo "Running tests for $TP_NUMBER for user $USERNAME..."
  bash "$TP_TEST_SCRIPT" "$USERNAME"
else
  echo "Error: Test script does not exist for $TP_NUMBER."
  exit 1
fi
