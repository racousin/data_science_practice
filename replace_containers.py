#!/usr/bin/env python3
"""
Step 2: Replace Bootstrap Container with Mantine Container usage
Since Container works similarly in both libraries, this is a safe first replacement
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

def replace_container_usage(content):
    """Replace Bootstrap Container usage patterns with Mantine equivalents"""
    
    # Skip files that don't have both Bootstrap and Mantine imports
    if 'react-bootstrap' not in content or '@mantine/core' not in content:
        return content, False
    
    original_content = content
    modified = False
    
    # Replace Container usage - these should work the same in both
    # Just need to ensure we're using the right import
    
    # Check if Container is in Bootstrap imports and remove it
    def remove_container_from_bootstrap_import(match):
        nonlocal modified
        imports = match.group(1)
        import_list = [item.strip() for item in imports.split(',')]
        
        # Remove Container from Bootstrap imports
        new_imports = [imp for imp in import_list if imp.strip() != 'Container']
        
        if len(new_imports) != len(import_list):
            modified = True
            if new_imports:
                return f"import {{ {', '.join(new_imports)} }} from 'react-bootstrap';"
            else:
                return ""  # Remove the entire import if Container was the only import
        
        return match.group(0)
    
    # Remove Container from Bootstrap imports
    bootstrap_import_pattern = r'import\s*{([^}]+)}\s*from\s*["\']react-bootstrap["\'];?'
    content = re.sub(bootstrap_import_pattern, remove_container_from_bootstrap_import, content)
    
    # Clean up empty import lines
    content = re.sub(r'\nimport\s*{\s*}\s*from\s*["\']react-bootstrap["\'];\n', '\n', content)
    content = re.sub(r'^import\s*{\s*}\s*from\s*["\']react-bootstrap["\'];\n', '', content, flags=re.MULTILINE)
    
    return content, modified

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content, modified = replace_container_usage(content)
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False

def main():
    print("Step 2: Replacing Bootstrap Container with Mantine Container...")
    
    js_files = find_js_jsx_files("website/src")
    print(f"Found {len(js_files)} JavaScript/JSX files")
    
    modified_files = []
    for file_path in js_files:
        if process_file(file_path):
            modified_files.append(file_path)
            print(f"Modified: {file_path}")
    
    print(f"\nProcessed {len(js_files)} files")
    print(f"Modified {len(modified_files)} files")

if __name__ == "__main__":
    main()