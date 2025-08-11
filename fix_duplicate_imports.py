#!/usr/bin/env python3
"""
Fix duplicate imports in Mantine import statements
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

def fix_duplicate_imports(content):
    """Fix duplicate imports in Mantine import statements"""
    
    def fix_mantine_import(match):
        imports = match.group(1)
        # Split imports and remove duplicates while preserving order
        import_list = [item.strip() for item in imports.split(',')]
        seen = set()
        unique_imports = []
        for imp in import_list:
            if imp and imp not in seen:
                seen.add(imp)
                unique_imports.append(imp)
        
        return f"import {{ {', '.join(unique_imports)} }} from '@mantine/core';"
    
    # Fix Mantine imports
    mantine_import_pattern = r'import\s*{([^}]+)}\s*from\s*["\']@mantine/core["\'];?'
    content = re.sub(mantine_import_pattern, fix_mantine_import, content)
    
    return content

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if '@mantine/core' not in content:
            return False
        
        original_content = content
        content = fix_duplicate_imports(content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False

def main():
    print("Fixing duplicate imports in Mantine import statements...")
    
    js_files = find_js_jsx_files("website/src")
    print(f"Found {len(js_files)} JavaScript/JSX files")
    
    modified_files = []
    for file_path in js_files:
        if process_file(file_path):
            modified_files.append(file_path)
            print(f"Fixed: {file_path}")
    
    print(f"\nProcessed {len(js_files)} files")
    print(f"Fixed {len(modified_files)} files")

if __name__ == "__main__":
    main()