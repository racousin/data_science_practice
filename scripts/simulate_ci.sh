#!/bin/bash

FETCH_ALL_MODULES=true

# Get the folder name from the current working directory as GITHUB_REPOSITORY_NAME
GITHUB_REPOSITORY_NAME=$(basename "$PWD")
export GITHUB_REPOSITORY_NAME

# Function to list all directories excluding the specified ones
list_authors() {
    for dir in $(ls -d */ | grep -vE 'venv/|scripts/|tests/|README.md'); do
        echo "Processing author directory: ${dir}"
        # Remove trailing slash from folder name to set as AUTHOR
        AUTHOR=$(basename "$dir")
        export AUTHOR

        # Simulate "Check and Add User to Students List"
        ./scripts/check-and-add-user.sh $AUTHOR $GITHUB_REPOSITORY_NAME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION

        # Simulate "Determine Changed Modules"
        export CHANGED_MODULES=$(./scripts/check-changed-modules.sh $AUTHOR $FETCH_ALL_MODULES)

        # Simulate "Run Tests and Update Results"
        if [[ -z "$CHANGED_MODULES" ]]; then
            echo "No MODULEs changed for $AUTHOR. Skipping tests."
        else
            ./tests/run_tests_and_update_results.sh $AUTHOR $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION "$CHANGED_MODULES" $GITHUB_REPOSITORY_NAME
            ./scripts/compute_progress_percentage.sh $AUTHOR $GITHUB_REPOSITORY_NAME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION
        fi
    done
}

# # Set permissions
chmod +x ./scripts/* ./tests/*/exercise*.sh ./tests/run_tests_and_update_results.sh

# Loop through all author directories
list_authors
