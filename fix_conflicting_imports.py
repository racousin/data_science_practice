#!/usr/bin/env python3
"""
Fix conflicting imports between Bootstrap and Mantine
Remove components from Bootstrap imports if they're also imported from Mantine
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

def fix_conflicting_imports(content):
    """Remove conflicting components from Bootstrap imports if they're also in Mantine"""
    
    # Skip files that don't have both imports
    if 'react-bootstrap' not in content or '@mantine/core' not in content:
        return content, False
    
    # Extract Mantine imports
    mantine_imports = set()
    mantine_import_pattern = r'import\s*{([^}]+)}\s*from\s*["\']@mantine/core["\'];?'
    mantine_matches = re.findall(mantine_import_pattern, content)
    for match in mantine_matches:
        imports = [item.strip() for item in match.split(',')]
        mantine_imports.update(imports)
    
    modified = False
    
    # Remove conflicting imports from Bootstrap
    def remove_conflicting_from_bootstrap(match):
        nonlocal modified
        imports = match.group(1)
        import_list = [item.strip() for item in imports.split(',')]
        
        # Remove imports that are also in Mantine
        new_imports = [imp for imp in import_list if imp.strip() not in mantine_imports]
        
        if len(new_imports) != len(import_list):
            modified = True
            if new_imports:
                return f"import {{ {', '.join(new_imports)} }} from 'react-bootstrap';"
            else:
                return ""  # Remove entire import if no components left
        
        return match.group(0)
    
    # Process Bootstrap imports
    bootstrap_import_pattern = r'import\s*{([^}]+)}\s*from\s*["\']react-bootstrap["\'];?'
    content = re.sub(bootstrap_import_pattern, remove_conflicting_from_bootstrap, content)
    
    # Clean up empty lines and empty imports
    content = re.sub(r'\n\s*\n', '\n', content)  # Remove multiple empty lines
    content = re.sub(r'import\s*{\s*}\s*from\s*["\']react-bootstrap["\'];\s*\n', '', content)
    
    return content, modified

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content, modified = fix_conflicting_imports(content)
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False

def main():
    print("Fixing conflicting imports between Bootstrap and Mantine...")
    
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