#!/bin/bash

# This script runs tests for tp3 and expects a username and AWS credentials
# Usage: ./run_tests.sh <username> <aws-access-key-id> <aws-secret-access-key> <aws-region>

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <username> <aws-access-key-id> <aws-secret-access-key> <aws-region>"
  exit 1
fi

USERNAME=$1
AWS_ACCESS_KEY_ID=$2
AWS_SECRET_ACCESS_KEY=$3
AWS_DEFAULT_REGION=$4
TP_NUMBER="3"
PREDICTIONS_PATH="${USERNAME}/tp${TP_NUMBER}/predictions.csv"
RESULTS_PATH="y_test.csv"

# Setup a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install boto3 pandas scikit-learn

# Download y_test.csv from S3
python tests/tp3/download_from_s3.py $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION

# # Run comparison
python tests/tp3/compare_results.py $RESULTS_PATH $PREDICTIONS_PATH

# # Deactivate the virtual environment
deactivate
