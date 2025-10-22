#!/usr/bin/env python3
"""
Script to update students.json with project information from project.csv
"""

import json
import csv

# File paths
STUDENTS_JSON = '/Users/raphaelcousin/data_science_practice/students.json'
PROJECT_CSV = '/Users/raphaelcousin/data_science_practice/project.csv'

def main():
    # Read students.json
    print("Reading students.json...")
    with open(STUDENTS_JSON, 'r', encoding='utf-8') as f:
        students = json.load(f)

    # Read project.csv
    print("Reading project.csv...")
    projects_by_github = {}
    with open(PROJECT_CSV, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            github_username = row['github']
            projects_by_github[github_username] = {
                'partner': row['partner'],
                'repo': row['repo'],
                'projet': row['projet'],
                'racousin_invited': row['invited']
            }

    # Update students with project information
    print("Updating students with project information...")
    updated_count = 0
    for github_username, student_data in students.items():
        if github_username in projects_by_github:
            student_data['project'] = projects_by_github[github_username]
            updated_count += 1
            print(f"  Updated {github_username}")
        else:
            print(f"  Warning: No project data found for {github_username}")

    # Write updated students.json
    print(f"\nWriting updated students.json...")
    with open(STUDENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(students, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Updated {updated_count} out of {len(students)} students.")

if __name__ == '__main__':
    main()
