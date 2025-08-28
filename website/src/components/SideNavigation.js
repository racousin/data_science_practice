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

// Course content links for modules with hierarchical structure including sublinks
export const courseContentData = {
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
export const exerciseContentData = {
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

// PyTorch course content links for modules with hierarchical structure
export const pytorchCourseContentData = {
  'module1': [
    { to: '/introduction', label: 'Part 1: Historical Context & Applications', subLinks: [
      { id: 'introduction', label: 'Introduction to Deep Learning' },
      { id: 'history', label: 'Historical Evolution' },
      { id: 'applications', label: 'Real-World Applications' },
      { id: 'data', label: 'Data: The Fuel of Deep Learning' }
    ]},
    { to: '/mathematical-framework', label: 'Part 2: Mathematical Framework', subLinks: [
      { id: 'ml-objective', label: 'The Machine Learning Objective' },
      { id: 'models-parameters', label: 'Models and Parameters' },
      { id: 'loss-functions', label: 'Loss Functions and Optimization' },
      { id: 'gradient-descent', label: 'Gradient Descent' },
      { id: 'linear-algebra', label: 'Essential Linear Algebra' }
    ]},
    { to: '/mlp-fundamentals', label: 'Part 3: Multi-Layer Perceptron Fundamentals', subLinks: [
      { id: 'neuron', label: 'Neuron as Computational Unit' },
      { id: 'network-architecture', label: 'Network Architecture' },
      { id: 'parameters', label: 'Parameters to Optimize' },
      { id: 'implementation', label: 'Complete Implementation' }
    ]},
    { to: '/pytorch-overview', label: 'Part 4: Deep Learning Frameworks', subLinks: [
      { id: 'pytorch-intro', label: 'PyTorch Overview' },
      { id: 'tensors', label: 'Tensors: The Foundation' },
      { id: 'autograd', label: 'Automatic Differentiation' },
      { id: 'ecosystem', label: 'PyTorch Ecosystem' },
      { id: 'resources', label: 'Resources and Best Practices' }
    ]}
  ],
  'module2': [
    { to: '/autograd-deep-dive', label: 'Autograd Deep Dive', subLinks: [
      { id: 'forward-reverse-mode', label: 'Forward & Reverse Mode Differentiation' },
      { id: 'computational-graph-construction', label: 'Computational Graph Construction' },
      { id: 'chain-rule-backpropagation', label: 'Chain Rule & Backpropagation Mathematics' },
      { id: 'gradient-accumulation', label: 'Gradient Accumulation & Zeroing' }
    ]},
    { to: '/advanced-gradient-mechanics', label: 'Advanced Gradient Mechanics', subLinks: [
      { id: 'gradient-flow', label: 'Gradient Flow & Vanishing/Exploding' },
      { id: 'gradient-clipping', label: 'Gradient Clipping & Normalization' },
      { id: 'higher-order-derivatives', label: 'Higher-order Derivatives & Hessians' },
      { id: 'custom-backward-passes', label: 'Custom Backward Passes' }
    ]},
    { to: '/optimization-algorithms', label: 'Optimization Algorithms', subLinks: [
      { id: 'modern-optimizers', label: 'Mathematical Foundations of Modern Optimizers' },
      { id: 'adam-rmsprop-adagrad', label: 'Adam, RMSprop, AdaGrad Derivations' },
      { id: 'learning-rate-scheduling', label: 'Learning Rate Scheduling Strategies' },
      { id: 'second-order-optimization', label: 'Second-order Optimization Methods' }
    ]}
  ],
  'module3': [
    { to: '/mlp-architecture-components', label: 'MLP Architecture & Components', subLinks: [
      { id: 'multilayer-perceptron', label: 'Multilayer Perceptron Mathematics' },
      { id: 'universal-approximation', label: 'Universal Approximation Theorem' },
      { id: 'activation-functions', label: 'Activation Functions: Mathematical Properties' },
      { id: 'weight-initialization', label: 'Weight Initialization Theory' },
      { id: 'regularization-techniques', label: 'Regularization Techniques (Dropout, L2, Batch Norm)' }
    ]},
    { to: '/data-pipeline-training-loop', label: 'Data Pipeline & Training Loop', subLinks: [
      { id: 'dataloader-architecture', label: 'DataLoader Architecture & Multiprocessing' },
      { id: 'batch-sampling', label: 'Batch Sampling Strategies' },
      { id: 'training-dynamics', label: 'Training Dynamics & Loss Landscapes' },
      { id: 'early-stopping', label: 'Early Stopping & Convergence Criteria' }
    ]},
    { to: '/monitoring-visualization', label: 'Monitoring & Visualization', subLinks: [
      { id: 'tensorboard-integration', label: 'TensorBoard Integration' },
      { id: 'metrics-visualization', label: 'Metrics Visualization Strategies' },
      { id: 'model-interpretability', label: 'Model Interpretability Basics' },
      { id: 'debugging-networks', label: 'Debugging Neural Networks' },
      { id: 'checkpoint-saving', label: 'Checkpoint Saving/Loading Strategies' }
    ]}
  ],
  'module4': [
    { to: '/device-management-resources', label: 'Device Management & Resources', subLinks: [
      { id: 'gpu-architecture', label: 'GPU Architecture for Deep Learning' },
      { id: 'memory-management', label: 'Memory Management Strategies' },
      { id: 'flops-memory-calculation', label: 'Calculate FLOPs & Memory Requirements' },
      { id: 'mixed-precision', label: 'Mixed Precision Training Mathematics' }
    ]},
    { to: '/model-optimization', label: 'Model Optimization', subLinks: [
      { id: 'model-compression', label: 'Model Compression Techniques' },
      { id: 'computational-complexity', label: 'Computational Complexity Analysis' },
      { id: 'torchscript-serialization', label: 'TorchScript & Model Serialization' },
      { id: 'jit-compilation', label: 'JIT Compilation Basics' }
    ]},
    { to: '/advanced-pytorch-architecture', label: 'Advanced PyTorch & Architecture Overview', subLinks: [
      { id: 'hooks-applications', label: 'Hooks & Their Applications' },
      { id: 'dynamic-computation-graphs', label: 'Dynamic Computation Graphs' },
      { id: 'cnn-convolution-mathematics', label: 'CNN Convolution Mathematics (Brief)' },
      { id: 'attention-mechanism-mathematics', label: 'Attention Mechanism Mathematics (Brief)' },
      { id: 'custom-cpp-extensions', label: 'Custom C++ Extensions Overview' }
    ]}
  ]
};

// PyTorch exercise content data
export const pytorchExerciseContentData = {
  'module1': [
    { to: '/exercise1', label: 'Exercise 1.1: Environment & Basics' },
    { to: '/exercise2', label: 'Exercise 1.2: Mathematical Implementation' },
    { to: '/exercise3', label: 'Exercise 1.3: Tensor Mastery' }
  ],
  'module2': [
    { to: '/exercise1', label: 'Exercise 2.1: Autograd Exploration' },
    { to: '/exercise2', label: 'Exercise 2.2: Gradient Analysis' },
    { to: '/exercise3', label: 'Exercise 2.3: Optimizer Implementation' }
  ],
  'module3': [
    { to: '/exercise1', label: 'Exercise 3.1: Build Custom MLP' },
    { to: '/exercise2', label: 'Exercise 3.2: Complete Training Pipeline' },
    { to: '/exercise3', label: 'Exercise 3.3: Data & Optimization' }
  ],
  'module4': [
    { to: '/exercise1', label: 'Exercise 4.1: Performance Profiling' },
    { to: '/exercise2', label: 'Exercise 4.2: Advanced Features' },
    { to: '/exercise3', label: 'Exercise 4.3: Mini-Project' }
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

// Course structure data with icons for modules
export const coursesData = {
  'python-deep-learning': {
    name: 'Python for Deep Learning (PyTorch)',
    modules: [
      { id: 'module1', name: 'Foundations of Deep Learning', icon: IconMathFunction },
      { id: 'module2', name: 'Automatic Differentiation & Optimization', icon: IconCpu },
      { id: 'module3', name: 'Neural Networks & Training Infrastructure', icon: IconBrain },
      { id: 'module4', name: 'Performance Optimization & Advanced Features', icon: IconCode },
    ]
  },
  'data-science-practice': {
    name: 'Data Science Practice',
    modules: [
      { id: 'module0', name: 'Prerequisites & Methodology', icon: IconChartBar },
      { id: 'module1', name: 'Git and Github', icon: IconGitBranch },
      { id: 'module2', name: 'Python Environment', icon: IconBrandPython },
      { id: 'module3', name: 'Data Science Landscape', icon: IconChartBar },
      { id: 'module4', name: 'Data Collection', icon: IconDatabase },
      { id: 'module5', name: 'Data Preprocessing', icon: IconFilter },
      { id: 'module6', name: 'Tabular Models', icon: IconTable },
      { id: 'module7', name: 'Deep Learning Fundamentals', icon: IconBrain },
      { id: 'module8', name: 'Image Processing', icon: IconPhoto },
      { id: 'module9', name: 'TimeSeries Processing', icon: IconClock },
      { id: 'module10', name: 'Natural Language Processing', icon: IconFileText },
      { id: 'module11', name: 'Generative Models', icon: IconSparkles },
      { id: 'module12', name: 'Recommendation Systems', icon: IconUsers },
      { id: 'module13', name: 'Reinforcement Learning', icon: IconRobot },
      { id: 'module14', name: 'Docker', icon: IconBrandDocker },
      { id: 'module15', name: 'Cloud Integration', icon: IconCloud },
      { id: 'project', name: 'Project' }
    ]
  }
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
                  ) : (
                    <NavLink
                      label={module.id === 'project' ? "Go to Project" : "Getting Started"}
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