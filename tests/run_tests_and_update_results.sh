#!/bin/bash

# This script runs tests for each TP and updates the user-specific JSON file with the results.

USERNAME=$1
AWS_ACCESS_KEY_ID=$2
AWS_SECRET_ACCESS_KEY=$3
AWS_DEFAULT_REGION=$4

# Download the current test results file
aws s3 cp "s3://www.raphaelcousin.com/students/${USERNAME}.json" "${USERNAME}.json"

for TP in $(ls tests | grep tp); do
    TP_NUMBER=$(echo $TP | cut -d'p' -f2)
    TP_TEST_SCRIPT="./tests/$TP/run_tests.sh"

    if [ -f "$TP_TEST_SCRIPT" ]; then
        echo "Running tests for $TP for user $USERNAME..."
        TEST_SUCCESS=$(bash "$TP_TEST_SCRIPT" "$USERNAME" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$AWS_DEFAULT_REGION" && echo "success" || echo "error")
        jq --arg tp "$TP_NUMBER" --arg result "$TEST_SUCCESS" '.[$tp] = $result' "${USERNAME}.json" > temp.json && mv temp.json "${USERNAME}.json"
    else
        echo "Error: Test script does not exist for $TP."
        exit 1
    fi
done

# Upload the updated test results file
aws s3 cp "${USERNAME}.json" "s3://www.raphaelcousin.com/students/${USERNAME}.json"
