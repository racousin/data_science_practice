#!/usr/bin/env python3
"""
Script to rename misnamed module files in the data-science-practice directories.
Fixes files where the module number in the filename doesn't match the directory module number.
"""

import os
import re
from pathlib import Path
import argparse

def fix_module_names(base_path, dry_run=True):
    """
    Fix module names in files that have incorrect module numbers.
    
    Args:
        base_path: Path to the website/public/modules/data-science-practice directory
        dry_run: If True, only show what would be renamed without actually renaming
    """
    
    # Modules to check
    modules_to_check = [7, 8, 9]
    
    # Track all renames
    renames = []
    
    for module_num in modules_to_check:
        module_dir = Path(base_path) / f"module{module_num}"
        
        if not module_dir.exists():
            print(f"Warning: {module_dir} does not exist")
            continue
            
        # Find all files that start with "module" and have .html or .ipynb extension
        patterns = ["module*.html", "module*.ipynb"]
        
        for pattern in patterns:
            for file_path in module_dir.rglob(pattern):
                filename = file_path.name
                
                # Extract the module number from the filename
                match = re.match(r'module(\d+)(_.*)', filename)
                if match:
                    file_module_num = int(match.group(1))
                    rest_of_filename = match.group(2)
                    
                    # Check if the module number in filename doesn't match the directory
                    if file_module_num != module_num:
                        # Create new filename with correct module number
                        new_filename = f"module{module_num}{rest_of_filename}"
                        new_path = file_path.parent / new_filename
                        
                        renames.append((file_path, new_path))
    
    # Process renames
    if not renames:
        print("No files need to be renamed.")
        return
    
    print(f"{'DRY RUN: ' if dry_run else ''}Found {len(renames)} file(s) to rename:\n")
    
    for old_path, new_path in renames:
        # Get relative paths for cleaner output
        try:
            old_rel = old_path.relative_to(Path.cwd())
            new_rel = new_path.relative_to(Path.cwd())
        except ValueError:
            old_rel = old_path
            new_rel = new_path
            
        print(f"  {old_rel}")
        print(f"  -> {new_rel}")
        
        if not dry_run:
            if new_path.exists():
                print(f"    WARNING: Target file already exists, skipping: {new_rel}")
            else:
                old_path.rename(new_path)
                print(f"    ✓ Renamed")
        print()
    
    if dry_run:
        print(f"\nTo actually rename these files, run with --execute flag")
    else:
        print(f"\n✓ Successfully renamed {len(renames)} file(s)")

def main():
    parser = argparse.ArgumentParser(
        description="Fix module names in data-science-practice files"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename files (default is dry-run mode)"
    )
    parser.add_argument(
        "--path",
        default="website/public/modules/data-science-practice",
        help="Base path to the modules directory"
    )
    
    args = parser.parse_args()
    
    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        return 1
    
    fix_module_names(base_path, dry_run=not args.execute)
    return 0

if __name__ == "__main__":
    exit(main())