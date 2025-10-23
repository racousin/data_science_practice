import React, { useState, useEffect } from 'react';
import { NavLink, Accordion, Text, Box, Divider, ScrollArea, Collapse } from '@mantine/core';
import { useLocation, Link } from 'react-router-dom';
import { 
  IconBook, 
  IconClipboardList, 
  IconChartBar, 
  IconChevronDown, 
  IconChevronRight,
  IconGitBranch,
  IconBrandPython,
  IconDatabase,
  IconFilter,
  IconTable,
  IconBrain,
  IconPhoto,
  IconClock,
  IconFileText,
  IconSparkles,
  IconUsers,
  IconRobot,
  IconBrandDocker,
  IconCloud,
  IconMathFunction,
  IconCpu,
  IconCode
} from '@tabler/icons-react';

// Project pages links
export const projectContentData = {
  'data-science-practice': [
    { to: '/2025', label: 'Project 2025' },
    { to: '/permuted-mnist', label: 'Option A: Permuted MNIST' },
    { to: '/bipedal-walker', label: 'Option B: Bipedal Walker' }
  ]
};

// Course content links for modules with hierarchical structure including sublinks
export const courseContentData = {
  'module1': [
    { to: '/Introduction', label: 'Introduction'},
    { to: '/installing-git', label: 'Installing Git'},
    { to: '/first-steps-with-git', label: 'First Steps with Git' },
    { to: '/configure-and-access-github', label: 'Configure GitHub' },
    { to: '/working-with-remote-repositories', label: 'Remote Repositories' },
    { to: '/branching-and-merging', label: 'Branching & Merging' },
    { to: '/collaborating', label: 'Collaborating' },
    { to: '/cheatsheet', label: 'Cheatsheet' },
        { to: '/git-hub-desktop', label: 'GitHub Desktop'},
            { to: '/git-hub-actions', label: 'GitHub Actions'}
  ],
  'module2': [
    { to: '/Introduction', label: 'Introduction' },
    { to: '/install-python', label: 'Install Python' },
    { to: '/setting-up-python-environment', label: 'Environment Setup' },
    { to: '/installing-packages', label: 'Installing Packages' },
    { to: '/ide', label: 'IDE' },
    { to: '/building-packages', label: 'Building Packages' },
    { to: '/syntax-and-linting', label: 'Building Packages' },
    { to: '/unit-and-integration-tests', label: 'Unit And Integration Tests' }
  ],
  'module3': [
    { to: '/Introduction', label: 'Introduction', subLinks: [
    ]},
    { to: '/machine-learning-pipeline', label: 'ML Pipeline', subLinks: [
    ]},
    { to: '/models-objective', label: 'Models Training & Prediction', subLinks: [
    ]},
    
        { to: '/evaluation-metrics', label: 'Evaluation Metrics', subLinks: [
    ]},

    { to: '/model-evaluation-validation', label: 'Model Evaluation', subLinks: [
    ]},

    { to: '/case-study', label: 'EDA Case Study' },
    { to: '/case-study-ml', label: 'Model Baseline Case Study' }
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
    { to: '/custom-objectives-guide', label: 'Custom Objectives' },
    { to: '/case-study', label: 'Case Study' }
  ],
  'module7': [
    { to: '/Introduction', label: 'Introduction' },
    { to: '/cnn-essentials', label: 'CNN Essentials' },
    { to: '/transfer-learning', label: 'Transfer Learning' },
    { to: '/object-detection', label: 'Object Detection' },
    { to: '/segmentation', label: 'Segmentation' },
    { to: '/generative-model', label: 'Generative Models' },
    { to: '/three-d-cnn', label: '3D CNN' },
    { to: '/enhancement', label: 'Enhancement Techniques' },
    { to: '/case-study', label: 'Case Study' }
  ],
  'module8': [
    { to: '/introduction', label: 'Introduction' },
    { to: '/tokenization', label: 'Tokenization' },
    { to: '/metrics-and-loss', label: 'Metrics and Loss' },
    { to: '/recurrent-networks', label: 'Recurrent Networks' },
    { to: '/attention-layer', label: 'Attention Layer' },
    { to: '/transformer', label: 'Transformer' },
    { to: '/training-transformers', label: 'Training Transformers' },
    { to: '/llm-transfer-learning', label: 'LLM and Transfer Learning' },
    { to: '/embeddings', label: 'Embeddings' },
    { to: '/rag', label: 'RAG' },
    { to: '/agentic', label: 'Agentic' }
  ],
  'module9': [
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
export const exerciseContentData = {
  'module1': [
    { to: '/exercise1', label: <span>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></span> },
    { to: '/exercise2', label: 'Exercise 2' },
    { to: '/exercise3', label: 'Exercise 3' },
    { to: '/exercise4', label: 'Exercise 4' },
    { to: '/exercise5', label: 'Exercise 5' }
    
    ],
  'module2': [
    { to: '/exercise1', label: <span>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></span> },
    { to: '/exercise2', label: 'Exercise 2' },
    { to: '/exercise3', label: 'Exercise 3' },
    { to: '/exercise4', label: 'Exercise 4' },
  ],
  'module3': [
    { to: '/exercise0', label: 'Exercise 0' },
    { to: '/exercise1', label: <span>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></span> },
    { to: '/exercise2', label: 'Exercise 2' }
  ],
  'module4': [
    { to: '/exercise1', label: <span>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></span> },
    { to: '/exercise2', label: 'Exercise 2' }
  ],
  'module5': [
    { to: '/exercise1', label: <span>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></span> },
  ],
  'module6': [
    { to: '/exercise1', label: <span>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></span> },
  ],
  'module7': [
    { to: '/exercise0', label: 'Exercise 0' },
    { to: '/exercise1', label: <span>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></span> },
    { to: '/exercise2', label: 'Exercise 2' }
  ],
  'module8': [
    { to: '/exercise0', label: 'Exercise 0' },
    { to: '/exercise1', label: 'Exercise 1' },
    { to: '/exercise2', label: 'Exercise 2' },
    { to: '/exercise3', label: <span>Exercise 3<span style={{color: 'red', fontWeight: 'bold'}}>*</span></span> },
  ],
  'module9': [
    { to: '/exercise0', label: 'Exercise 0' },
    { to: '/exercise1', label: 'Exercise 1' },
    { to: '/exercise2', label: <span>Exercise 2<span style={{color: 'red', fontWeight: 'bold'}}>*</span></span> },
    { to: '/exercise3', label: 'Exercise 3' },
    { to: '/exercise4', label: 'Exercise 4' },
    { to: '/exercise5', label: 'Exercise 5' },
    { to: '/exercise6', label: 'Exercise 6' }
  ]
};

// PyTorch course content links for modules with hierarchical structure
export const pytorchCourseContentData = {
  'module1': [
    { to: '/introduction', label: 'Historical Context & Applications', subLinks: [
    ]},
    { to: '/mathematical-framework', label: 'Machine Learning in a nutshell', subLinks: [
    ]},
    { to: '/mlp-fundamentals', label: 'Multi layer perceptron in a nutshell', subLinks: [
    ]},
    { to: '/pytorch-introduction', label: 'PyTorch: Introduction', subLinks: [
    ]},
    { to: '/pytorch-mlp-fundamentals', label: 'PyTorch MLP Fundamentals', subLinks: [
    ]}
  ],
  'module2': [
    { to: '/automatic-differentiation-mathematical-perspective', label: 'Autograd mathematical perspective', subLinks: [
    ]},
    { to: '/autograd-torch-perspective', label: 'Autograd torch perspective', subLinks: [
    ]},
    { to: '/py-torch-optimizers', label: 'PyTorch Optimizers', subLinks: [
    ]},
    { to: '/advanced-gradient-mechanics', label: 'Understanding Gradient Flow', subLinks: [
    ]}
  ],
  'module3': [
    { to: '/training-basics', label: 'Training Basics', subLinks: [
    ]},
    { to: '/essential-layers', label: 'NN Essential Components', subLinks: [
    ]},
    { to: '/data-pipeline-training-loop', label: 'Data Pipeline Essential Components', subLinks: [
    ]},
    { to: '/monitoring-visualization', label: 'Monitoring & Visualization', subLinks: [
    ]}
  ],
  'module4': [
    { to: '/resource-profiling', label: 'Model Resource Profiling', subLinks: [

    ]},
    { to: '/performance-optimization', label: 'Performance Optimization', subLinks: [
    ]},
        { to: '/fine-tuning', label: 'Fine-Tuning', subLinks: [
    ]},
    { to: '/multi-gpu-scaling', label: 'Multi-GPU Scaling', subLinks: [
    ]}
  ]
};

// PyTorch exercise content data
export const pytorchExerciseContentData = {
  'module1': [
    { to: '/exercise1', label: 'Exercise 1.1: Environment & Basics' },
    { to: '/exercise2', label: 'Exercise 1.2: Gradient Descent' },
    { to: '/exercise3', label: 'Exercise 1.3: First Step with MLP' }
  ],
  'module2': [
    { to: '/exercise1', label: 'Exercise 2.1: Autograd Exploration' },
    { to: '/exercise2', label: 'Exercise 2.2: Optimization with PyTorch Autograd' },
    { to: '/exercise3', label: 'Exercise 2.3: Gradient Flow' }
  ],
  'module3': [
    { to: '/exercise0', label: 'Exercise 3.0: Training Basic' },
    { to: '/exercise1', label: 'Exercise 3.1: Data Pipeline & Training Loop' },
    { to: '/exercise2', label: 'Exercise 3.2: Essential Layers' },
    { to: '/exercise3', label: 'Exercise 3.3: Monitoring & Visualization with TensorBoard' }
  ],
  'module4': [
    { to: '/exercise1', label: 'Exercise 4.1: Model Resource Profiling' },
    { to: '/exercise2', label: 'Exercise 4.2: Fine Tunning' },
    { to: '/exercise3', label: 'Exercise 4.3: Model Resource optimization' },
  ]
};

// Function to get course content links for a module
export const getCourseContentLinks = (moduleId, courseId) => {
  if (courseId === 'data-science-practice' && courseContentData[moduleId]) {
    return courseContentData[moduleId];
  }
  if (courseId === 'python-deep-learning' && pytorchCourseContentData[moduleId]) {
    return pytorchCourseContentData[moduleId];
  }
  return [];
};

// Function to get exercise links for a module
export const getExerciseContentLinks = (moduleId, courseId) => {
  if (courseId === 'data-science-practice' && exerciseContentData[moduleId]) {
    return exerciseContentData[moduleId];
  }
  if (courseId === 'python-deep-learning' && pytorchExerciseContentData[moduleId]) {
    return pytorchExerciseContentData[moduleId];
  }
  return [];
};

// Function to get project links
export const getProjectContentLinks = (courseId) => {
  if (projectContentData[courseId]) {
    return projectContentData[courseId];
  }
  return [];
};

// Course structure data with icons for modules
export const coursesData = {
  'python-deep-learning': {
    name: 'Python for Deep Learning (PyTorch)',
    modules: [
      { id: 'module1', name: 'Foundations of Deep Learning', icon: IconMathFunction },
      { id: 'module2', name: 'Automatic Differentiation', icon: IconCpu },
      { id: 'module3', name: 'Neural Network Training & Monitoring', icon: IconBrain },
      { id: 'module4', name: 'Performance Optimization & Scale', icon: IconCode },
    ]
  },
  'data-science-practice': {
    name: 'Data Science Practice',
    modules: [
      { id: 'module0', name: 'Prerequisites & Methodology', icon: IconChartBar },
      { id: 'module1', name: 'Git and Github', icon: IconGitBranch },
      { id: 'module2', name: 'Python Environment', icon: IconBrandPython },
      { id: 'module3', name: 'Data Science Methodology', icon: IconChartBar },
      { id: 'module4', name: 'Data Collection', icon: IconDatabase },
      { id: 'module5', name: 'Data Preprocessing', icon: IconFilter },
      { id: 'module6', name: 'Tabular Models', icon: IconTable },
      { id: 'module7', name: 'Image Processing', icon: IconPhoto },
      { id: 'module8', name: 'Natural Language Processing', icon: IconFileText },
      { id: 'module9', name: 'Reinforcement Learning', icon: IconRobot },
      { id: 'project', name: 'Project' }
    ]
  }
};

// Project content section
const ProjectContentSection = ({ projectContentLinks, currentCourseId, currentPath, onClose }) => {
  return (
    <Box pl="md" mt="xs">
      {projectContentLinks.map((link, index) => (
        <NavLink
          key={index}
          label={link.label}
          component={Link}
          to={`/courses/${currentCourseId}/project${link.to}`}
          active={currentPath.includes(`/courses/${currentCourseId}/project${link.to}`)}
          pl="xs"
          size="sm"
          styles={{ label: { fontSize: '0.85rem' } }}
          onClick={onClose}
        />
      ))}
    </Box>
  );
};

// Course content section with hierarchical sublinks
const CourseContentSection = ({ courseContentLinks, currentCourseId, moduleId, currentPath, onClose }) => {
  // Initialize open sections based on current path
  const [openSections, setOpenSections] = useState(() => {
    const initial = {};
    courseContentLinks.forEach((link, index) => {
      if (currentPath.includes(`/course${link.to}`)) {
        initial[index] = true;
      }
    });
    return initial;
  });

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
            onClick={onClose}
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
const ExerciseContentSection = ({ exerciseContentLinks, currentCourseId, moduleId, currentPath, onClose }) => {
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
          onClick={onClose}
        />
      ))}
    </Box>
  );
};

const SideNavigation = ({ onClose }) => {
  const location = useLocation();
  const path = location.pathname;
  
  // Extract course ID from the path: /courses/[courseId]/...
  const pathParts = path.split('/').filter(Boolean);
  const inCoursesSection = pathParts[0] === 'courses';
  
  const currentCourseId = pathParts.length > 1 ? pathParts[1] : null;
  const currentModuleId = pathParts.length > 2 ? pathParts[2] : null;
  
  // State for accordion - must be declared before any returns
  const [expandedModule, setExpandedModule] = useState(currentModuleId);
  
  // Update expanded module when navigation changes
  useEffect(() => {
    if (currentModuleId) {
      setExpandedModule(currentModuleId);
    }
  }, [currentModuleId]);
  
  // If not in courses section or on the courses index page, don't show the side nav
  if (!inCoursesSection || pathParts.length <= 1) return null;
  
  const courseInfo = coursesData[currentCourseId];
  
  if (!courseInfo) return null;
  
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
          onClick={onClose}
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
            onClick={onClose}
          />
        )}
        
        <Divider my="sm" />
        
        {/* Modules accordion */}
        <Accordion value={expandedModule} onChange={setExpandedModule} variant="filled">
          {courseInfo.modules.map(module => {
            const isCurrentModule = currentModuleId === module.id;
            const courseContentLinks = getCourseContentLinks(module.id, currentCourseId);
            const exerciseContentLinks = getExerciseContentLinks(module.id, currentCourseId);
            const projectContentLinks = module.id === 'project' ? getProjectContentLinks(currentCourseId) : [];

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
                        active={path === `/courses/${currentCourseId}/${module.id}/course`}
                        pl="xs"
                        icon={<IconBook size={16} />}
                        onClick={onClose}
                      />

                      {/* Course Content Links - show when in current module */}
                      {isCurrentModule && courseContentLinks.length > 0 && (
                        <CourseContentSection
                          courseContentLinks={courseContentLinks}
                          currentCourseId={currentCourseId}
                          moduleId={module.id}
                          currentPath={path}
                          onClose={onClose}
                        />
                      )}

                      <NavLink
                        label="Exercise Overview"
                        component={Link}
                        to={`/courses/${currentCourseId}/${module.id}/exercise`}
                        active={path === `/courses/${currentCourseId}/${module.id}/exercise`}
                        pl="xs"
                        icon={<IconClipboardList size={16} />}
                        onClick={onClose}
                      />

                      {/* Exercise Content Links - show when in current module */}
                      {isCurrentModule && exerciseContentLinks.length > 0 && (
                        <ExerciseContentSection
                          exerciseContentLinks={exerciseContentLinks}
                          currentCourseId={currentCourseId}
                          moduleId={module.id}
                          currentPath={path}
                          onClose={onClose}
                        />
                      )}
                    </>
                  ) : module.id === 'project' ? (
                    <>
                      <NavLink
                        label="Project Overview"
                        component={Link}
                        to={`/courses/${currentCourseId}/${module.id}`}
                        active={path === `/courses/${currentCourseId}/${module.id}` && !path.includes('/project/2')}
                        pl="xs"
                        icon={<IconBook size={16} />}
                        onClick={onClose}
                      />

                      {/* Project Content Links - always show for project module */}
                      {projectContentLinks.length > 0 && (
                        <ProjectContentSection
                          projectContentLinks={projectContentLinks}
                          currentCourseId={currentCourseId}
                          currentPath={path}
                          onClose={onClose}
                        />
                      )}
                    </>
                  ) : (
                    <NavLink
                      label="Getting Started"
                      component={Link}
                      to={`/courses/${currentCourseId}/${module.id}`}
                      active={path.includes(`/courses/${currentCourseId}/${module.id}`)}
                      pl="xs"
                      icon={<IconBook size={16} />}
                      onClick={onClose}
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

// Export function to get module index from module id
export const getModuleIndex = (moduleId) => {
  if (!moduleId || moduleId === 'project') return null;
  const match = moduleId.match(/module(\d+)/);
  return match ? parseInt(match[1]) : null;
};

export default SideNavigation;