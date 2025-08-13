# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal website and teaching platform featuring:
- **Personal Portfolio**: Home page with profile, teaching, and research/projects sections
- **Multi-Course Teaching Platform**: Supporting multiple data science and deep learning courses
- **Python Testing Framework**: Automated assessment system at `/tests`
- **Content Management**: Course materials and Jupyter notebooks

## Design Principles

### Cultural Direction
- **Academic and Professional**: Maintain a scholarly, educational focus
- **Concise and Direct**: Clear communication without unnecessary elaboration
- **Objective and Neutral**: Factual, impersonal presentation of content
- **Non-commercial**: Avoid marketing language, focus on educational value

### Visual Style
- **Minimalist**: Simple, clean interfaces without clutter
- **Sober Colors**: Modern, professional palette avoiding bright or playful colors
- **Direct Interaction**: Clickable elements without unnecessary buttons or decorations
- **Functional Design**: Every element serves a clear purpose

### Content Guidelines
- **Simple Navigation**: Direct links and lists over complex cards or widgets
- **Clear Typography**: Readable fonts with appropriate hierarchy
- **Dark Mode Support**: Accessible viewing in different lighting conditions
- **Responsive Layout**: Works seamlessly across devices

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

# Deploy to AWS (build, upload to S3, create CloudFront invalidation)
./scripts/deploy.sh
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
pip install -r tests/data-science-practice/requirements.txt

# Run tests for specific module (example for module 1)
./tests/data-science-practice/module1/exercise1.sh <username> <timestamp> <aws_key> <aws_secret> <aws_region>
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
2. **Python Deep Learning**: 4 modules on PyTorch and deep learning (under development)

### Navigation Features
- **Hierarchical Sidebar**: Course content with expandable subsections
- **Exercise Integration**: Direct links to exercises in sidebar
- **No Duplicate Menus**: All navigation centralized in left sidebar for course pages

### Testing System

#### Data Science Practice Course
Students work with a dedicated Git repository workflow:
- Create and push solutions to their own GitHub repository
- Run shell scripts (`exercise*.sh`) to validate their solutions
- Test results are automatically uploaded to AWS S3 for instructor tracking
- Progress is monitored via the Session Results page

#### Python Deep Learning Course  
Students work directly in Google Colab notebooks:
- Open exercise notebooks in Colab environment
- Clone test repository (https://github.com/racousin/data_science_practice.git) for step-by-step validation
- Run pytest tests for each exercise part to verify their solutions
- No separate repository or submission required
- Immediate feedback with validation throughout exercises

### Content Structure
Each module contains:
- Course materials (theory)
- Exercises (practice)
- Jupyter notebooks (interactive coding)

Notebooks are converted to HTML via the Makefile for web display.

## Important Path File Resources

### Course Content Structure
```
/website/public/modules/
├── data-science-practice/
│   ├── module1/
│   │   ├── course.ipynb
│   │   ├── course.html
│   │   └── exercises/
│   └── ...module15/
└── python-deep-learning/
    ├── module1/
    │   ├── course.ipynb
    │   ├── course.html
    │   └── exercises/
    └── ...module4/
```

### Source Code Organization
```
/website/src/
├── components/
│   ├── MainHeader.js         # Top navigation
│   ├── SideNavigation.js     # Course sidebar navigation (defines course structure)
│   └── ModuleFrame.js        # Module page wrapper
├── pages/
│   ├── Home.js               # Landing page
│   ├── Courses.js            # Course selection
│   ├── Projects.js           # Research projects
│   ├── data-science-practice/
│   │   └── module{1-15}/
│   │       ├── course/
│   │       └── exercise/
│   └── python-deep-learning/
│       └── module{1-4}/
│           ├── course/
│           └── exercise/
└── courses/
    ├── DataSciencePractice.js
    └── PythonDeepLearning.js
```

### Testing Framework
```
/tests/
├── data-science-practice/        # Repository-based workflow
│   ├── requirements.txt
│   └── module{1-15}/
│       ├── exercise{N}.sh        # Shell script runner (uploads to S3)
│       └── test_exercise{N}.py   # Pytest test file
└── python-deep-learning/          # Colab self-validation workflow
    ├── requirements.txt
    └── module{1-4}/
        └── test_exercise{N}.py   # Pytest test file (no .sh scripts)
```

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
- `/courses/python-deep-learning/module{N}/course` - Course content
- `/courses/python-deep-learning/module{N}/exercise` - Exercises
- `/projects` - Research projects and collaborations