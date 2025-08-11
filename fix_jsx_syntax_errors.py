#!/usr/bin/env python3
"""
Fix specific JSX syntax errors in the codebase.
Targets the most common error patterns identified:
1. Mismatched JSX tags (e.g., <tr> vs </Table.Tr>)
2. Double closing tags (e.g., </Text> </Text>)
3. Invalid syntax patterns (e.g., </Stack>),;)
4. Malformed Text content
"""

import os
import re
import sys

def fix_jsx_syntax_errors(content):
    """Fix JSX syntax errors in content"""
    
    # Pattern 1: Fix mismatched table row tags
    # Replace <tr> with <Table.Tr> when followed by </Table.Tr>
    content = re.sub(r'<tr(\s[^>]*)?>', r'<Table.Tr\1>', content)
    
    # Pattern 2: Fix double closing tags (specifically </Text> </Text>)
    content = re.sub(r'</Text>\s*</Text>', '</Text>', content)
    
    # Pattern 3: Fix invalid syntax like </Stack>),; 
    content = re.sub(r'</Stack>\),;', '</Stack>', content)
    
    # Pattern 4: Fix malformed Text tags like <Text>content</Text>}</Text>
    content = re.sub(r'(<Text[^>]*>.*?</Text>)\}(</Text>)', r'\1', content)
    
    # Pattern 5: Fix incomplete Text tags like <Text>content that got cut off
    # Look for lines ending with incomplete text content
    content = re.sub(r'(<Text[^>]*>[^<]*?)\n\s*</Text>\s*</Text>', r'\1</Text>', content, flags=re.MULTILINE)
    
    # Pattern 6: Fix stray closing </Text> after proper closing
    content = re.sub(r'(</Text>)\s*</Text>', r'\1', content)
    
    # Pattern 7: Fix cases where Text content is malformed with }
    content = re.sub(r'<Text>([^<]*?)</Text>\}</Text>', r'<Text>\1</Text>', content)
    
    # Pattern 8: Fix incomplete sentences in Text tags (reconnect broken lines)
    # This handles cases where content got split mid-sentence
    content = re.sub(r'(<Text[^>]*>[^<]*?)\n\s*</Text>\s*</Text>', r'\1</Text>', content, flags=re.MULTILINE)
    
    # Pattern 9: Fix Weight attribute (should be fw in Mantine)
    content = re.sub(r'<Text\s+weight=\{([^}]+)\}>', r'<Text fw={\1}>', content)
    
    return content

def process_file(file_path):
    """Process a single JavaScript/JSX file"""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Apply fixes
        fixed_content = fix_jsx_syntax_errors(original_content)
        
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
    print("Processing files...")
    
    fixed_count = 0
    processed_count = 0
    
    for file_path in js_files:
        if process_file(file_path):
            fixed_count += 1
        processed_count += 1
        
        # Progress update every 25 files
        if processed_count % 25 == 0:
            print(f"Progress: {processed_count}/{len(js_files)} files processed")
    
    print(f"\nCompleted processing {len(js_files)} files")
    print(f"Fixed: {fixed_count} files")
    print(f"No changes needed: {len(js_files) - fixed_count} files")

if __name__ == "__main__":
    main()