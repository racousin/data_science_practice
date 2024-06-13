#!/bin/bash
# Computes the progress percentage based on passed exercises

USER=$1
GITHUB_REPOSITORY_NAME=$2
AWS_ACCESS_KEY_ID=$3
AWS_SECRET_ACCESS_KEY=$4
AWS_DEFAULT_REGION=$5

# Configure AWS CLI
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_DEFAULT_REGION

# Download the student's JSON file
aws s3 cp s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/"$USER".json "$USER".json

# Calculate the progress percentage
TOTAL_EXERCISES=$(jq '.[] | .[] | .is_passed_test' "$USER".json | wc -l)
PASSED_EXERCISES=$(jq '.[] | .[] | select(.is_passed_test == true) | .is_passed_test' "$USER".json | wc -l)

if [ "$TOTAL_EXERCISES" -eq 0 ]; then
  PROGRESS=0
else
  PROGRESS=$(echo "scale=2; $PASSED_EXERCISES * 100 / $TOTAL_EXERCISES" | bc)
fi

# Update the progress in the JSON file
jq --argjson progress "$PROGRESS" '.progress_percentage = $progress' "$USER".json > temp.json && mv temp.json "$USER".json

# Upload the updated JSON file to S3
aws s3 cp "$USER".json s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/"$USER".json
