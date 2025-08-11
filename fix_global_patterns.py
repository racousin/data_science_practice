#!/usr/bin/env python3
"""
Fix well-understood global patterns from the Bootstrap to Mantine migration.
These are patterns we've identified that consistently cause syntax errors.
"""

import os
import re
import sys

def fix_global_patterns(content):
    """Fix well-understood global patterns"""
    
    # Pattern 1: Fix missing return statement closing - very common pattern
    # Functions that end with </Stack>}; or similar instead of </Stack>);
    content = re.sub(r'(\s+)(</[A-Z][a-zA-Z.]*>)\s*\};', r'\1\2\n  );\n};', content)
    
    # Pattern 2: Fix ),; at end of component returns - very common
    content = re.sub(r'(\s+)(</[A-Z][a-zA-Z.]*>)\),;', r'\1\2', content)
    
    # Pattern 3: Fix double closing Text tags - well understood pattern
    content = re.sub(r'</Text>\s*</Text>', '</Text>', content)
    
    # Pattern 4: Fix incomplete Text content where line breaks cut off closing tags
    # Pattern: <Text>content that continues on next line
    #          </Text>
    content = re.sub(r'(<Text[^>]*>[^<]*?)\n\s*</Text>', r'\1</Text>', content, flags=re.MULTILINE)
    
    # Pattern 5: Fix Boxlapse -> Collapse (typo we found)
    content = re.sub(r'<Boxlapse', '<Collapse', content)
    content = re.sub(r'</Boxlapse>', '</Collapse>', content)
    
    # Pattern 6: Fix weight -> fw attribute in Mantine Text components
    content = re.sub(r'<Text\s+weight=\{([^}]+)\}', r'<Text fw={\1}', content)
    
    # Pattern 7: Fix stray ),; at end of lines
    content = re.sub(r'\),;\s*$', '', content, flags=re.MULTILINE)
    
    # Pattern 8: Fix incomplete component closures in Stack/Box components
    content = re.sub(r'(\s+)(</Stack>)\s*\)\s*;\s*$', r'\1\2', content, flags=re.MULTILINE)
    
    return content

def process_file(file_path):
    """Process a single JavaScript/JSX file"""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Apply fixes
        fixed_content = fix_global_patterns(original_content)
        
        # Only write if changes were made
        if fixed_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            print(f"✓ Fixed: {file_path}")
            return True
        else:
            print(f"- No changes: {file_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path}: {str(e)}")
        return False

def main():
    """Main function to process all JS/JSX files"""
    
    # Start from website directory
    root_dir = "/Users/raphaelcousin/data_science_practice/website/src"
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist")
        sys.exit(1)
    
    # Find all JS/JSX files
    js_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.js', '.jsx')):
                js_files.append(os.path.join(root, file))
    
    print(f"Found {len(js_files)} JavaScript/JSX files")
    print("Applying well-understood global pattern fixes...")
    
    fixed_count = 0
    processed_count = 0
    
    for file_path in js_files:
        if process_file(file_path):
            fixed_count += 1
        processed_count += 1
        
        # Progress update every 50 files
        if processed_count % 50 == 0:
            print(f"Progress: {processed_count}/{len(js_files)} files processed")
    
    print(f"\nCompleted processing {len(js_files)} files")
    print(f"Fixed: {fixed_count} files with global pattern issues")
    print(f"No changes needed: {len(js_files) - fixed_count} files")

if __name__ == "__main__":
    main()