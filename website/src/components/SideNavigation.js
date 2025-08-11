import React, { useState } from 'react';
import { NavLink, Accordion, Text, Box, Divider, ScrollArea, Collapse } from '@mantine/core';
import { useLocation, Link } from 'react-router-dom';
import { IconBook, IconClipboardList, IconChartBar, IconChevronDown, IconChevronRight } from '@tabler/icons-react';

// Course content links for modules with hierarchical structure including sublinks
const courseContentData = {
  'module1': [
    { to: '/Introduction', label: 'Introduction', subLinks: [
      { id: 'what-is-version-control', label: 'What is Version Control?' },
      { id: 'git-vs-other-vcs', label: 'Git vs Other VCS' },
      { id: 'git-terminology', label: 'Git Terminology' }
    ]},
    { to: '/installing-git', label: 'Installing Git', subLinks: [
      { id: 'windows', label: 'Windows Installation' },
      { id: 'mac', label: 'Mac Installation' },
      { id: 'linux', label: 'Linux Installation' }
    ]},
    { to: '/first-steps-with-git', label: 'First Steps with Git' },
    { to: '/configure-and-access-github', label: 'Configure GitHub' },
    { to: '/working-with-remote-repositories', label: 'Remote Repositories' },
    { to: '/branching-and-merging', label: 'Branching & Merging' },
    { to: '/collaborating', label: 'Collaborating' },
    { to: '/best-practices-and-resources', label: 'Best Practices' }
  ],
  'module2': [
    { to: '/Introduction', label: 'Introduction' },
    { to: '/install-python', label: 'Install Python' },
    { to: '/setting-up-python-environment', label: 'Environment Setup' },
    { to: '/installing-packages', label: 'Installing Packages' },
    { to: '/building-packages', label: 'Building Packages' },
    { to: '/best-practices-and-ressources', label: 'Best Practices' }
  ],
  'module3': [
    { to: '/Introduction', label: 'Introduction', subLinks: [
      { id: 'data', label: 'The Data' },
      { id: 'applications', label: 'The Applications' },
      { id: 'roles', label: 'Roles in Data Science' },
      { id: 'tools', label: 'The Data Science Tools' }
    ]},
    { to: '/machine-learning-pipeline', label: 'ML Pipeline', subLinks: [
      { id: 'problem-definition', label: 'Problem Definition' },
      { id: 'data-collection', label: 'Data Collection' },
      { id: 'data-cleaning', label: 'Data Preprocessing and Feature Engineering' },
      { id: 'model-building', label: 'Model Selection, Training, and Evaluation' },
      { id: 'deployment', label: 'Deployment, Monitoring, and Maintenance' },
      { id: 'monitoring', label: 'Model Interpretability and Explainability' },
      { id: 'best-practices', label: 'Best Practices: Baseline and Iterate' }
    ]},
    { to: '/model-training-prediction', label: 'Model Training & Prediction', subLinks: [
      { id: 'model-fitting', label: 'Training' },
      { id: 'prediction', label: 'Prediction' }
    ]},
    { to: '/model-evaluation-validation', label: 'Model Evaluation', subLinks: [
      { id: 'performance-metrics', label: 'Performance Metrics' },
      { id: 'overfitting-underfitting', label: 'Overfitting and Underfitting' },
      { id: 'bias-variance', label: 'Bias-Variance Tradeoff' },
      { id: 'cross-validation', label: 'Cross-Validation' },
      { id: 'time-series-cv', label: 'Time Series Cross-Validation' }
    ]},
    { to: '/evaluation-metrics', label: 'Evaluation Metrics', subLinks: [
      { id: 'regression-metrics', label: 'Regression Metrics' },
      { id: 'binary-classification-metrics', label: 'Binary Classification Metrics' },
      { id: 'multi-class-classification-metrics', label: 'Multi-class Classification Metrics' },
      { id: 'ranking-metrics', label: 'Ranking Metrics' },
      { id: 'time-series-metrics', label: 'Time Series Metrics' },
      { id: 'choosing-metrics', label: 'Choosing the Right Metric' }
    ]},
    { to: '/exploratory-data-analysis', label: 'Exploratory Data Analysis', subLinks: [
      { id: 'main-components', label: 'Main Components of EDA' },
      { id: 'jupyter-notebooks', label: 'Jupyter Notebooks' },
      { id: 'google-colab', label: 'Google Colab' }
    ]},
    { to: '/eda-case-study', label: 'EDA Case Study' },
    { to: '/model-baseline-case-study', label: 'Model Baseline Case Study' }
  ],
  'module4': [
    { to: '/Introduction', label: 'Introduction' },
    { to: '/files', label: 'Files' },
    { to: '/databases', label: 'Databases' },
    { to: '/apis', label: 'APIs' },
    { to: '/web-scraping', label: 'Web Scraping' },
    { to: '/batch-vs-streaming', label: 'Batch vs Streaming' },
    { to: '/data-quality', label: 'Data Quality' },
    { to: '/manipulating-sources', label: 'Manipulating Sources' },
    { to: '/case-study', label: 'Case Study' }
  ],
  'module5': [
    { to: '/Introduction', label: 'Introduction' },
    { to: '/handle-inconsistencies', label: 'Handle Inconsistencies' },
    { to: '/handle-duplicates', label: 'Handle Duplicates' },
    { to: '/handle-missing-values', label: 'Handle Missing Values' },
    { to: '/handle-categorical-values', label: 'Handle Categorical Values' },
    { to: '/handle-outliers', label: 'Handle Outliers' },
    { to: '/feature-engineering', label: 'Feature Engineering' },
    { to: '/scaling-and-normalization', label: 'Scaling & Normalization' },
    { to: '/feature-selection-and-dimensionality-reduction', label: 'Feature Selection' }
  ],
  'module6': [
    { to: '/model-selection', label: 'Model Selection' },
    { to: '/hyperparameter-optimization', label: 'Hyperparameter Optimization' },
    { to: '/models', label: 'Models' },
    { to: '/ensemble-models', label: 'Ensemble Models' },
    { to: '/ensemble-techniques', label: 'Ensemble Techniques' },
    { to: '/time-series-models', label: 'Time Series Models' },
    { to: '/automl', label: 'AutoML' },
    { to: '/custom-objectives-guide', label: 'Custom Objectives' },
    { to: '/case-study', label: 'Case Study' }
  ],
  'module7': [
    { to: '/Introduction', label: 'Introduction' },
    { to: '/backpropagation', label: 'Backpropagation' },
    { to: '/essential-components', label: 'Essential Components' },
    { to: '/nn-workflow', label: 'NN Workflow' },
    { to: '/case-study', label: 'Case Study' }
  ],
  'module8': [
    { to: '/Introduction', label: 'Introduction' },
    { to: '/cnn-essentials', label: 'CNN Essentials' },
    { to: '/transfer-learning', label: 'Transfer Learning' },
    { to: '/enhancement', label: 'Enhancement Techniques' },
    { to: '/case-study', label: 'Case Study' }
  ],
  'module10': [
    { to: '/Introduction', label: 'Introduction to NLP' },
    { to: '/text-numerical-representation', label: 'Text Numerical Representation' },
    { to: '/rnn', label: 'RNN' },
    { to: '/transformer-components', label: 'Transformer Components' },
    { to: '/transformer-architectures', label: 'Transformer Architectures' },
    { to: '/transfer-learning', label: 'Transfer Learning' },
    { to: '/nlp-evaluation', label: 'NLP Evaluation' }
  ],
  'module13': [
    { to: '/Introduction', label: 'Introduction' },
    { to: '/mdp', label: 'Markov Decision Processes' },
    { to: '/dynamic-programming', label: 'Dynamic Programming' },
    { to: '/rl-paradigms', label: 'RL Paradigms' },
    { to: '/model-free-methods', label: 'Model-Free Methods' },
    { to: '/deep-model-free', label: 'Deep Model-Free' },
    { to: '/rl-training-efficiency', label: 'RL Training Efficiency' }
  ]
};

// Exercise links for modules that have exercises
const exerciseContentData = {
  'module1': [
    { to: '/exercise1', label: 'Exercise 1' },
    { to: '/exercise2', label: 'Exercise 2' }
  ],
  'module2': [
    { to: '/exercise1', label: 'Exercise 1' },
    { to: '/exercise2', label: 'Exercise 2' }
  ],
  'module3': [
    { to: '/exercise0', label: 'Exercise 0' },
    { to: '/exercise1', label: 'Exercise 1' },
    { to: '/exercise2', label: 'Exercise 2' }
  ],
  'module4': [
    { to: '/exercise1', label: 'Exercise 1' }
  ],
  'module5': [
    { to: '/exercise1', label: 'Exercise 1' },
    { to: '/exercise2', label: 'Exercise 2' }
  ],
  'module6': [
    { to: '/exercise1', label: 'Exercise 1' },
    { to: '/exercise2', label: 'Exercise 2' }
  ],
  'module7': [
    { to: '/exercise0', label: 'Exercise 0' },
    { to: '/exercise1', label: 'Exercise 1' }
  ],
  'module8': [
    { to: '/exercise0', label: 'Exercise 0' },
    { to: '/exercise1', label: 'Exercise 1' }
  ],
  'module10': [
    { to: '/exercise0', label: 'Exercise 0' },
    { to: '/exercise1', label: 'Exercise 1' },
    { to: '/exercise2', label: 'Exercise 2' },
    { to: '/exercise3', label: 'Exercise 3' }
  ],
  'module13': [
    { to: '/exercise0', label: 'Exercise 0' },
    { to: '/exercise1', label: 'Exercise 1' },
    { to: '/exercise2', label: 'Exercise 2' },
    { to: '/exercise3', label: 'Exercise 3' },
    { to: '/exercise4', label: 'Exercise 4' },
    { to: '/exercise5', label: 'Exercise 5' },
    { to: '/exercise6', label: 'Exercise 6' }
  ]
};

// Function to get course content links for a module
const getCourseContentLinks = (moduleId, courseId) => {
  if (courseId === 'data-science-practice' && courseContentData[moduleId]) {
    return courseContentData[moduleId];
  }
  return [];
};

// Function to get exercise links for a module
const getExerciseContentLinks = (moduleId, courseId) => {
  if (courseId === 'data-science-practice' && exerciseContentData[moduleId]) {
    return exerciseContentData[moduleId];
  }
  return [];
};

// Course structure data
const coursesData = {
  'data-science-practice': {
    name: 'Data Science Practice',
    modules: [
      { id: 'module0', name: 'Prerequisites & Methodology' },
      { id: 'module1', name: 'Git and Github' },
      { id: 'module2', name: 'Python Environment' },
      { id: 'module3', name: 'Data Science Landscape' },
      { id: 'module4', name: 'Data Collection' },
      { id: 'module5', name: 'Data Preprocessing' },
      { id: 'module6', name: 'Tabular Models' },
      { id: 'module7', name: 'Deep Learning Fundamentals' },
      { id: 'module8', name: 'Image Processing' },
      { id: 'module9', name: 'TimeSeries Processing' },
      { id: 'module10', name: 'Natural Language Processing' },
      { id: 'module11', name: 'Generative Models' },
      { id: 'module12', name: 'Recommendation Systems' },
      { id: 'module13', name: 'Reinforcement Learning' },
      { id: 'module14', name: 'Docker' },
      { id: 'module15', name: 'Cloud Integration' },
      { id: 'project', name: 'Project' }
    ]
  },
  'python-deep-learning': {
    name: 'Python for Deep Learning',
    modules: [
      { id: 'module1', name: 'Introduction to Tensors' },
      { id: 'module2', name: 'PyTorch Fundamentals' },
      { id: 'module3', name: 'TensorBoard Visualization' },
    ]
  }
};

// Course content section with hierarchical sublinks
const CourseContentSection = ({ courseContentLinks, currentCourseId, moduleId, currentPath }) => {
  const [openSections, setOpenSections] = useState({});

  const toggleSection = (index) => {
    setOpenSections(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return (
    <Box pl="md" mt="xs">
      {courseContentLinks.map((link, index) => (
        <Box key={index} mb="xs">
          <NavLink
            label={
              <Box style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                <span>{link.label}</span>
                {link.subLinks && link.subLinks.length > 0 && (
                  <Box
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      toggleSection(index);
                    }}
                    style={{ cursor: 'pointer' }}
                  >
                    {openSections[index] ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />}
                  </Box>
                )}
              </Box>
            }
            component={Link}
            to={`/courses/${currentCourseId}/${moduleId}/course${link.to}`}
            active={currentPath.includes(`/courses/${currentCourseId}/${moduleId}/course${link.to}`)}
            pl="xs"
            size="sm"
            styles={{ label: { fontSize: '0.85rem' } }}
          />
          
          {link.subLinks && (
            <Collapse in={openSections[index]}>
              <Box pl="lg" mt="xs">
                {link.subLinks.map((subLink, subIndex) => (
                  <NavLink
                    key={subIndex}
                    label={subLink.label}
                    component="a"
                    href={`/courses/${currentCourseId}/${moduleId}/course${link.to}#${subLink.id}`}
                    pl="xs"
                    size="xs"
                    styles={{ 
                      label: { fontSize: '0.75rem', color: 'var(--mantine-color-gray-6)' },
                      root: { marginBottom: '2px' }
                    }}
                  />
                ))}
              </Box>
            </Collapse>
          )}
        </Box>
      ))}
    </Box>
  );
};

// Exercise content section
const ExerciseContentSection = ({ exerciseContentLinks, currentCourseId, moduleId, currentPath }) => {
  return (
    <Box pl="md" mt="xs">
      {exerciseContentLinks.map((link, index) => (
        <NavLink
          key={index}
          label={link.label}
          component={Link}
          to={`/courses/${currentCourseId}/${moduleId}/exercise${link.to}`}
          active={currentPath.includes(`/courses/${currentCourseId}/${moduleId}/exercise${link.to}`)}
          pl="xs"
          size="sm"
          styles={{ label: { fontSize: '0.85rem' } }}
        />
      ))}
    </Box>
  );
};

const SideNavigation = () => {
  const location = useLocation();
  const path = location.pathname;
  
  // Extract course ID from the path: /courses/[courseId]/...
  const pathParts = path.split('/').filter(Boolean);
  const inCoursesSection = pathParts[0] === 'courses';
  
  // If not in courses section or on the courses index page, don't show the side nav
  if (!inCoursesSection || pathParts.length <= 1) return null;
  
  const currentCourseId = pathParts[1];
  const courseInfo = coursesData[currentCourseId];
  
  if (!courseInfo) return null;
  
  // Extract current module info
  const currentModuleId = pathParts.length > 2 ? pathParts[2] : null;
  
  return (
    <ScrollArea h="100%">
      <Box p="md">
        <Text fw={700} size="lg" mb="md">
          {courseInfo.name}
        </Text>
        
        {/* Back to course overview */}
        <NavLink
          label="Course Overview"
          component={Link}
          to={`/courses/${currentCourseId}`}
          active={path === `/courses/${currentCourseId}`}
          mb="md"
          icon={<IconBook size={16} />}
        />
        
        {/* Results link - only for data-science-practice for now */}
        {currentCourseId === 'data-science-practice' && (
          <NavLink
            label="Session Results"
            component={Link}
            to={`/courses/${currentCourseId}/results`}
            active={path.includes(`/courses/${currentCourseId}/results`)}
            mb="md"
            icon={<IconChartBar size={16} />}
          />
        )}
        
        <Divider my="sm" />
        
        {/* Modules accordion */}
        <Accordion defaultValue={currentModuleId} variant="filled">
          {courseInfo.modules.map(module => {
            const isCurrentModule = currentModuleId === module.id;
            const courseContentLinks = getCourseContentLinks(module.id, currentCourseId);
            const exerciseContentLinks = getExerciseContentLinks(module.id, currentCourseId);
            
            return (
              <Accordion.Item key={module.id} value={module.id}>
                <Accordion.Control>{module.name}</Accordion.Control>
                <Accordion.Panel>
                  {module.id !== 'module0' && module.id !== 'project' ? (
                    <>
                      <NavLink
                        label="Course Overview"
                        component={Link}
                        to={`/courses/${currentCourseId}/${module.id}/course`}
                        active={path === `/courses/${currentCourseId}/${module.id}/course` && !path.includes('/course/')}
                        pl="xs"
                        icon={<IconBook size={16} />}
                      />
                      
                      {/* Course Content Links - show when in current module */}
                      {isCurrentModule && courseContentLinks.length > 0 && (
                        <CourseContentSection 
                          courseContentLinks={courseContentLinks}
                          currentCourseId={currentCourseId}
                          moduleId={module.id}
                          currentPath={path}
                        />
                      )}
                      
                      <NavLink
                        label="Exercise Overview"
                        component={Link}
                        to={`/courses/${currentCourseId}/${module.id}/exercise`}
                        active={path === `/courses/${currentCourseId}/${module.id}/exercise` && !path.includes('/exercise/')}
                        pl="xs"
                        icon={<IconClipboardList size={16} />}
                      />
                      
                      {/* Exercise Content Links - show when in current module */}
                      {isCurrentModule && exerciseContentLinks.length > 0 && (
                        <ExerciseContentSection 
                          exerciseContentLinks={exerciseContentLinks}
                          currentCourseId={currentCourseId}
                          moduleId={module.id}
                          currentPath={path}
                        />
                      )}
                    </>
                  ) : (
                    <NavLink
                      label={module.id === 'project' ? "Go to Project" : "Getting Started"}
                      component={Link}
                      to={`/courses/${currentCourseId}/${module.id}`}
                      active={path.includes(`/courses/${currentCourseId}/${module.id}`)}
                      pl="xs"
                      icon={<IconBook size={16} />}
                    />
                  )}
                </Accordion.Panel>
              </Accordion.Item>
            );
          })}
        </Accordion>
      </Box>
    </ScrollArea>
  );
};

export default SideNavigation;