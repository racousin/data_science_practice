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

def add_grid_to_mantine_import(content):
    """Add Grid to Mantine imports if Grid is used but not imported"""
    # Check if Grid is used in the file
    if '<Grid' not in content and '<Grid.Col' not in content:
        return content, False
    
    # Check if Grid is already in Mantine imports
    mantine_import_pattern = r'import\s*{([^}]+)}\s*from\s*["\']@mantine/core["\'];?'
    match = re.search(mantine_import_pattern, content)
    
    if not match:
        return content, False
    
    imports = match.group(1)
    if 'Grid' in imports:
        return content, False  # Grid already imported
    
    # Add Grid to the imports
    imports_list = [item.strip() for item in imports.split(',')]
    imports_list.append('Grid')
    new_imports = ', '.join(imports_list)
    
    new_import = f"import {{ {new_imports} }} from '@mantine/core';"
    content = re.sub(mantine_import_pattern, new_import, content)
    
    return content, True

def main():
    import sys
    
    # Check if we should add Grid to imports
    if len(sys.argv) > 1 and sys.argv[1] == "--add-grid":
        print("Adding Grid to Mantine imports where needed...")
        
        js_files = find_js_jsx_files("website/src")
        print(f"Found {len(js_files)} JavaScript/JSX files")
        
        modified_files = []
        for file_path in js_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content, modified = add_grid_to_mantine_import(content)
                
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    modified_files.append(file_path)
                    print(f"Added Grid import: {file_path}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"\nProcessed {len(js_files)} files")
        print(f"Updated {len(modified_files)} files")
        
    else:
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