#!/bin/bash
# This script runs tests for changed MODULES and updates the user-specific JSON file with the results.

USERNAME=$1
AWS_ACCESS_KEY_ID=$2
AWS_SECRET_ACCESS_KEY=$3
AWS_DEFAULT_REGION=$4
CHANGED_MODULES=$5
GITHUB_REPOSITORY_NAME=$6

# Configure AWS CLI
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_DEFAULT_REGION

# Check if the user's results file exists; if not, create an empty JSON object
if ! aws s3 ls "s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/${USERNAME}.json" > /dev/null; then
    echo "{}" > "${USERNAME}.json"
else
    aws s3 cp "s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/${USERNAME}.json" "${USERNAME}.json"
fi

# Iterate over each changed MODULE and run tests
for module in $CHANGED_MODULES; do
    MODULE_TEST_SCRIPT="./tests/module${module}/run_tests.sh"

    if [ -f "$MODULE_TEST_SCRIPT" ]; then
        echo "Running tests for module${module} for user ${USERNAME}..."
        TEST_SUCCESS=$(bash "$MODULE_TEST_SCRIPT" "$USERNAME" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$AWS_DEFAULT_REGION" && echo "success" || echo "error")

        # Update the JSON file with the test result
        EXERCISE_NAME="exercise$module" # Adjust this to the actual exercise name if needed
        jq --arg module "module$module" --arg exercise "$EXERCISE_NAME" --arg test_result "$TEST_SUCCESS" \
           '.[$module][$exercise] = {is_passed_test: ($test_result == "success"), score: "", logs: $test_result}' \
           "${USERNAME}.json" > temp.json && mv temp.json "${USERNAME}.json"
    else
        echo "Error: Test script does not exist for module${module}."
    fi
done

# Upload the updated test results file
aws s3 cp "${USERNAME}.json" "s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/${USERNAME}.json"
