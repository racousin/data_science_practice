#!/usr/bin/env python3
"""
Replace Bootstrap component usage with Mantine equivalents
This script does the component replacements while keeping the HTML structure intact
"""

import os
import re

def find_js_jsx_files(directory):
    js_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.js', '.jsx')):
                js_files.append(os.path.join(root, file))
    return js_files

def replace_bootstrap_components(content):
    """Replace Bootstrap JSX components with Mantine equivalents"""
    
    # Skip files that don't have react-bootstrap imports
    if 'react-bootstrap' not in content:
        return content, False
    
    modified = False
    
    # Start with a simple approach: just replace a few files manually first to test
    # We'll do this more carefully, testing one component type at a time
    
    # First, let's just replace Col components (simpler case)
    col_replacements = [
        # Handle Col with md prop
        (r'<Col\s+md=\{([^}]+)\}([^>]*)>', r'<Grid.Col span={{ md: \1 }}\2>'),
        # Handle Col with sm prop  
        (r'<Col\s+sm=\{([^}]+)\}([^>]*)>', r'<Grid.Col span={{ sm: \1 }}\2>'),
        # Handle Col with xs prop
        (r'<Col\s+xs=\{([^}]+)\}([^>]*)>', r'<Grid.Col span={{ xs: \1 }}\2>'),
        # Handle Col with lg prop
        (r'<Col\s+lg=\{([^}]+)\}([^>]*)>', r'<Grid.Col span={{ lg: \1 }}\2>'),
        # Handle plain Col
        (r'<Col([^>]*)>', r'<Grid.Col\1>'),
        # Handle closing Col tag
        (r'</Col>', '</Grid.Col>'),
    ]
    
    # Apply Col replacements first
    for old_pattern, new_pattern in col_replacements:
        new_content = re.sub(old_pattern, new_pattern, content)
        if new_content != content:
            modified = True
            content = new_content
    
    # Only replace Row after Col is done (to avoid nesting issues)
    if modified:  # Only if we had Col replacements
        row_replacements = [
            (r'<Row([^>]*)>', r'<Grid\1>'),
            (r'</Row>', '</Grid>'),
        ]
        
        for old_pattern, new_pattern in row_replacements:
            new_content = re.sub(old_pattern, new_pattern, content)
            if new_content != content:
                content = new_content
    
    return content, modified

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content, modified = replace_bootstrap_components(content)
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False

def main():
    import sys
    
    # Check if we should test on specific files first
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Testing Bootstrap component replacement on a few files...")
        
        # Just test with a few files first
        test_files = [
            "website/src/pages/module1/CourseGit.js",
            "website/src/pages/module1/ExerciseGit.js",
            "website/src/pages/module2/CoursePython.js",
        ]
        
        files_to_process = [f for f in test_files if os.path.exists(f)]
        print(f"Testing on {len(files_to_process)} files")
    else:
        print("Replacing Bootstrap components with Mantine equivalents...")
        files_to_process = find_js_jsx_files("website/src")
        print(f"Found {len(files_to_process)} JavaScript/JSX files")
    
    modified_files = []
    for file_path in files_to_process:
        if process_file(file_path):
            modified_files.append(file_path)
            print(f"Updated: {file_path}")
    
    print(f"\nProcessed {len(files_to_process)} files")
    print(f"Updated {len(modified_files)} files")

if __name__ == "__main__":
    main()