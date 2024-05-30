#!/bin/bash

# Usage: ./run_tests.sh <username> <tp-number> <aws-access-key-id> <aws-secret-access-key> <aws-region>
# Example: ./run_tests.sh alice tp1 key secret region

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <username> <tp-number> <aws-access-key-id> <aws-secret-access-key> <aws-region>"
  exit 1
fi


USERNAME=$1
TP_NUMBER=$2
AWS_ACCESS_KEY_ID=$3
AWS_SECRET_ACCESS_KEY=$4
AWS_DEFAULT_REGION=$5

TP_TEST_SCRIPT="./tests/$TP_NUMBER/run_tests.sh"

if [ -f "$TP_TEST_SCRIPT" ]; then
  echo "Running tests for $TP_NUMBER for user $USERNAME..."
  bash "$TP_TEST_SCRIPT" "$USERNAME" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$AWS_DEFAULT_REGION"
else
  echo "Error: Test script does not exist for $TP_NUMBER."
  exit 1
fi
