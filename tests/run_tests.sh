#!/bin/bash

# Usage: ./run_tests.sh <username> <module-number> <aws-access-key-id> <aws-secret-access-key> <aws-region>

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <username> <module-number> <aws-access-key-id> <aws-secret-access-key> <aws-region>"
  exit 1
fi


USERNAME=$1
MODULE_NUMBER=$2
AWS_ACCESS_KEY_ID=$3
AWS_SECRET_ACCESS_KEY=$4
AWS_DEFAULT_REGION=$5

MODULE_TEST_SCRIPT="./tests/$MODULE_NUMBER/run_tests.sh"

if [ -f "$MODULE_TEST_SCRIPT" ]; then
  echo "Running tests for $MODULE_NUMBER for user $USERNAME..."
  bash "$MODULE_TEST_SCRIPT" "$USERNAME" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$AWS_DEFAULT_REGION"
else
  echo "Error: Test script does not exist for $MODULE_NUMBER."
  exit 1
fi
