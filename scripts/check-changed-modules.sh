#!/bin/bash
# Script to determine changed modules

USER=$1
git diff --name-only HEAD^ HEAD | grep "^${USER}/module" | sed -E 's|.*/module([0-9]+)/.*|\1|' | tr '\n' ' ' | xargs
