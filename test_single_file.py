#!/usr/bin/env python3
"""
Test the replacement on a single file first
"""

import re

def replace_bootstrap_components(content):
    """Replace Bootstrap JSX components with Mantine equivalents"""
    
    # Skip files that don't have react-bootstrap imports
    if 'react-bootstrap' not in content:
        return content, False
    
    modified = False
    
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

def test_single_file():
    file_path = "website/src/pages/module1/CourseGit.js"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Original content (relevant lines):")
    lines = content.split('\n')
    for i, line in enumerate(lines[170:195], 171):
        print(f"{i}: {line}")
    
    new_content, modified = replace_bootstrap_components(content)
    
    if modified:
        print("\nModified content (relevant lines):")
        new_lines = new_content.split('\n')
        for i, line in enumerate(new_lines[170:195], 171):
            print(f"{i}: {line}")
        
        print(f"\nWould modify: {modified}")
    else:
        print("No changes needed")

if __name__ == "__main__":
    test_single_file()