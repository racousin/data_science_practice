#!/usr/bin/env python3
import re
import sys

def replace_section_tags(content):
    """Replace Section tags with Title tags."""
    # Pattern to match multi-line <Section> tags with attributes in any order
    # Matches opening <Section tag with any attributes including icon, title, and id
    pattern = r'<Section\s+[^>]*?title="([^"]+)"[^>]*?id="([^"]+)"[^>]*?>'
    
    # Replacement pattern
    replacement = r'<Title order={3} id="\2">\1</Title>'
    
    # Perform the replacement with DOTALL and MULTILINE flags
    result = re.sub(pattern, replacement, content, flags=re.DOTALL | re.MULTILINE)
    
    return result

def process_file(filepath):
    """Process a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified_content = replace_section_tags(content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"Processed: {filepath}")
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python replace_section_tags.py <file1> [file2] ...")
        print("Example: python replace_section_tags.py file.js")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        process_file(filepath)

if __name__ == "__main__":
    main()