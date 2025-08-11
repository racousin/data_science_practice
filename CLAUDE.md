# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal website and teaching platform featuring:
- **Personal Portfolio**: Home page with profile, teaching, and research/projects sections
- **Multi-Course Teaching Platform**: Supporting multiple data science and deep learning courses
- **Python Testing Framework**: Automated assessment system at `/tests`
- **Content Management**: Course materials and Jupyter notebooks

## Common Commands

### Website Development
```bash
# Install dependencies
cd website && npm install

# Start development server
cd website && npm start

# Build for production
cd website && npm run build

# Run tests
cd website && npm test
```

### Jupyter Notebook Processing
```bash
# Convert all module notebooks to HTML
make

# Clean generated HTML files
make clean
```

### Testing Framework
```bash
# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r tests/requirements.txt

# Run tests for specific module (example for module 1)
./tests/module1/exercise1.sh <username> <timestamp> <aws_key> <aws_secret> <aws_region>
```

## Architecture

### Site Structure
The website is organized with multiple layout types:
- **Simple Layout** (no sidebar): Home, Courses List, Projects pages
- **Course Layout** (with sidebar): Course content pages with hierarchical navigation

### Frontend (React)
The website uses React 18.3 with Mantine UI components. Key directories:
- `/website/src/pages/` - Page components (Home, Projects, Course module pages)
- `/website/src/courses/` - Course containers (DataSciencePractice, PythonDeepLearning)
- `/website/src/components/` - Reusable components:
  - `MainHeader` - Top navigation bar
  - `SideNavigation` - Course-specific sidebar with hierarchical content links
  - `ModuleFrame` - Module page wrapper
- `/website/public/modules/` - Module content (HTML, Jupyter notebooks)

### Supported Courses
1. **Data Science Practice**: 15 modules covering Git, Python, ML, Deep Learning, NLP, Computer Vision
2. **Python Deep Learning**: 3 modules on PyTorch and deep learning (under development)

### Navigation Features
- **Hierarchical Sidebar**: Course content with expandable subsections
- **Exercise Integration**: Direct links to exercises in sidebar
- **No Duplicate Menus**: All navigation centralized in left sidebar for course pages

### Testing System
Python-based automated testing with pytest. Each module has:
- Shell script runner (`exercise*.sh`)
- Python test file (`test_exercise*.py`)
- Results are uploaded to AWS S3 for progress tracking

### Content Structure
Each module contains:
- Course materials (theory)
- Exercises (practice)
- Jupyter notebooks (interactive coding)

Notebooks are converted to HTML via the Makefile for web display.

## Key Dependencies

### Frontend
- React Router for navigation
- Mantine UI for components
- react-jupyter for notebook display
- react-ga4 for analytics

### Testing
- pytest for test execution
- pandas/scikit-learn for data science operations
- boto3 for AWS S3 integration

## Development Notes

### Technical Notes
- Test results are stored in S3 bucket: `www.raphaelcousin.com/repositories/{repo}/students/`
- Virtual environment activation is required for Python operations
- All module content is stored as HTML or Jupyter notebooks in `/website/public/modules/`
- Course content links are defined in `/website/src/components/SideNavigation.js`

### URL Structure
- `/` - Home page (personal profile)
- `/courses` - Course selection page
- `/courses/data-science-practice/module{N}/course` - Course content
- `/courses/data-science-practice/module{N}/exercise` - Exercises
- `/courses/data-science-practice/results` - Student results
- `/projects` - Research projects and collaborations