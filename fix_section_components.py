#!/usr/bin/env python3
import re
import sys

def fix_section_components(content):
    """Replace Section component usage and remove the component definition."""
    
    # 1. Replace Section component usage with Title
    # Handle all 6 possible orders of icon, title, id attributes
    
    # Order 1: icon, title, id
    pattern1 = r'<Section[\s\S]*?icon=\{[^}]+\}[\s\S]*?title="([^"]+)"[\s\S]*?id="([^"]+)"[\s\S]*?>'
    content = re.sub(pattern1, r'<Title order={3} id="\2">\1</Title>', content, flags=re.DOTALL)
    
    # Order 2: icon, id, title
    pattern2 = r'<Section[\s\S]*?icon=\{[^}]+\}[\s\S]*?id="([^"]+)"[\s\S]*?title="([^"]+)"[\s\S]*?>'
    content = re.sub(pattern2, r'<Title order={3} id="\1">\2</Title>', content, flags=re.DOTALL)
    
    # Order 3: title, icon, id
    pattern3 = r'<Section[\s\S]*?title="([^"]+)"[\s\S]*?icon=\{[^}]+\}[\s\S]*?id="([^"]+)"[\s\S]*?>'
    content = re.sub(pattern3, r'<Title order={3} id="\2">\1</Title>', content, flags=re.DOTALL)
    
    # Order 4: title, id, icon
    pattern4 = r'<Section[\s\S]*?title="([^"]+)"[\s\S]*?id="([^"]+)"[\s\S]*?icon=\{[^}]+\}[\s\S]*?>'
    content = re.sub(pattern4, r'<Title order={3} id="\2">\1</Title>', content, flags=re.DOTALL)
    
    # Order 5: id, icon, title
    pattern5 = r'<Section[\s\S]*?id="([^"]+)"[\s\S]*?icon=\{[^}]+\}[\s\S]*?title="([^"]+)"[\s\S]*?>'
    content = re.sub(pattern5, r'<Title order={3} id="\1">\2</Title>', content, flags=re.DOTALL)
    
    # Order 6: id, title, icon
    pattern6 = r'<Section[\s\S]*?id="([^"]+)"[\s\S]*?title="([^"]+)"[\s\S]*?icon=\{[^}]+\}[\s\S]*?>'
    content = re.sub(pattern6, r'<Title order={3} id="\1">\2</Title>', content, flags=re.DOTALL)
    
    # 2. Remove the Section component definition
    # Pattern to match the Section component definition (including the entire function)
    section_def_pattern = r'const Section = \([^)]*\) => \([^;]*\);'
    content = re.sub(section_def_pattern, '', content, flags=re.DOTALL)
    
    # 3. Also remove any closing </Section> tags
    content = re.sub(r'</Section>', '', content)
    
    return content

def process_file(filepath):
    """Process a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process if file contains Section tags with icons
        if '<Section' in content:
            modified_content = fix_section_components(content)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"Fixed Section components in: {filepath}")
        else:
            print(f"No Section tags found in: {filepath}")
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    if len(sys.argv) < 2:
        # Process known files with Section components
        files_with_section = [
            'website/src/pages/data-science-practice/module6/course/CustomObjectivesGuide.js',
            'website/src/pages/data-science-practice/module6/course/HyperparameterOptimization.js',
            'website/src/pages/data-science-practice/module6/course/EnsembleTechniques.js',
            'website/src/pages/data-science-practice/module6/course/ModelSelection.js',
            'website/src/pages/data-science-practice/module5/course/FeatureSelectionAndDimensionalityReduction.js',
            'website/src/pages/data-science-practice/module5/course/FeatureEngineering.js',
            'website/src/pages/data-science-practice/module5/course/ScalingAndNormalization.js',
            'website/src/pages/data-science-practice/module5/course/HandleOutliers.js',
            'website/src/pages/data-science-practice/module5/course/HandleCategoricalValues.js',
            'website/src/pages/data-science-practice/module4/exercise/Exercise1.js',
            'website/src/pages/data-science-practice/module4/course/CaseStudy.js'
        ]
        for filepath in files_with_section:
            process_file(filepath)
    else:
        for filepath in sys.argv[1:]:
            process_file(filepath)

if __name__ == "__main__":
    main()