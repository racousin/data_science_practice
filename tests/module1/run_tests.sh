#!/bin/bash

# This script runs tests for TP1 and expects a username as a parameter
# Usage: ./run_tests.sh <username>

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <username> <aws-access-key-id> <aws-secret-access-key> <aws-region>"
  exit 1
fi

USERNAME=$1
MODULE_NUMBER="1"  # Since this script is specifically for TP1, we can hardcode the module number.

echo "Starting tests for TP1 for user $USERNAME..."

# Define the expected file path based on the parameters
FILE_PATH="${USERNAME}/module${MODULE_NUMBER}/user"

# Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
  echo "Error: File $FILE_PATH does not exist."
  exit 1
fi

# Check if the file contains the correct content
# Format should be 'username,surname,name'
FILE_CONTENT=$(cat "$FILE_PATH")
if [[ ! "$FILE_CONTENT" =~ ^${USERNAME},[a-zA-Z]+,[a-zA-Z]+$ ]]; then
  echo "Error: File content format is incorrect in $FILE_PATH."
  echo "Received: '$FILE_CONTENT'"
  echo "Expected format: '${USERNAME},surname,name'"
  exit 1
fi

echo "module${MODULE_NUMBER} tests passed successfully for ${USERNAME}."
