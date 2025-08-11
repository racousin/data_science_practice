#!/usr/bin/env python3
"""
Step 1: Add Mantine imports alongside Bootstrap imports
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

def add_mantine_imports(content):
    # Find Bootstrap imports
    bootstrap_import_pattern = r'import\s*{([^}]+)}\s*from\s*["\']react-bootstrap["\'];?'
    
    def process_bootstrap_import(match):
        imports = match.group(1)
        import_list = [item.strip() for item in imports.split(',')]
        
        # Map Bootstrap to Mantine components
        bootstrap_to_mantine = {
            'Container': 'Container',
            'Row': 'Grid',
            'Col': 'Grid',
            'Button': 'Button',
            'Card': 'Card',
            'Nav': 'Stack',
            'NavLink': 'NavLink',
            'Alert': 'Alert',
            'Modal': 'Modal',
            'Table': 'Table',
            'Badge': 'Badge',
            'Image': 'Image',
            'Accordion': 'Accordion',
        }
        
        # Get corresponding Mantine imports
        mantine_imports = set()
        for imp in import_list:
            if imp in bootstrap_to_mantine:
                mantine_imports.add(bootstrap_to_mantine[imp])
        
        # Keep original Bootstrap import and add Mantine import
        original_import = match.group(0)
        
        if mantine_imports:
            mantine_import = f"import {{ {', '.join(sorted(mantine_imports))} }} from '@mantine/core';"
            return f"{original_import}\n{mantine_import}"
        
        return original_import
    
    return re.sub(bootstrap_import_pattern, process_bootstrap_import, content)

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'react-bootstrap' not in content or '@mantine/core' in content:
            return False
        
        original_content = content
        content = add_mantine_imports(content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False

def main():
    print("Step 1: Adding Mantine imports alongside Bootstrap imports...")
    
    js_files = find_js_jsx_files("src")
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