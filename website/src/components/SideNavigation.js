import React, { useState } from 'react';
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
  IconCube,
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
    { to: '/Introduction', label: 'Introduction to PyTorch', subLinks: [
      { id: 'pytorch-ecosystem', label: 'PyTorch Ecosystem Overview' },
      { id: 'installation-setup', label: 'Installation & Environment Setup' },
      { id: 'first-tensor', label: 'Your First Tensor Operations' }
    ]},
    { to: '/tensor-fundamentals', label: 'Tensor Fundamentals', subLinks: [
      { id: 'tensor-creation', label: 'Tensor Creation Methods' },
      { id: 'data-types', label: 'Data Types & Precision' },
      { id: 'tensor-attributes', label: 'Tensor Attributes & Properties' },
      { id: 'indexing-slicing', label: 'Indexing & Slicing' }
    ]},
    { to: '/tensor-operations', label: 'Tensor Operations', subLinks: [
      { id: 'element-wise-ops', label: 'Element-wise Operations' },
      { id: 'reduction-ops', label: 'Reduction Operations' },
      { id: 'broadcasting', label: 'Broadcasting Mechanics' },
      { id: 'tensor-manipulation', label: 'Shape Manipulation' }
    ]},
    { to: '/linear-algebra', label: 'Linear Algebra Operations', subLinks: [
      { id: 'matrix-multiplication', label: 'Matrix Multiplication Variants' },
      { id: 'decompositions', label: 'Matrix Decompositions' },
      { id: 'eigenvalues', label: 'Eigenvalues & Eigenvectors' },
      { id: 'norms', label: 'Vector & Matrix Norms' }
    ]},
    { to: '/memory-management', label: 'Memory Management', subLinks: [
      { id: 'storage-views', label: 'Storage & Views' },
      { id: 'contiguous-tensors', label: 'Contiguous vs Non-contiguous' },
      { id: 'memory-layout', label: 'Memory Layout & Strides' },
      { id: 'in-place-operations', label: 'In-place Operations' }
    ]},
    { to: '/advanced-indexing', label: 'Advanced Indexing', subLinks: [
      { id: 'fancy-indexing', label: 'Fancy Indexing' },
      { id: 'boolean-indexing', label: 'Boolean Indexing' },
      { id: 'gather-scatter', label: 'Gather & Scatter Operations' },
      { id: 'masked-operations', label: 'Masked Operations' }
    ]}
  ],
  'module2': [
    { to: '/neural-network-basics', label: 'Neural Network Basics', subLinks: [
      { id: 'perceptron', label: 'The Perceptron' },
      { id: 'multilayer', label: 'Multilayer Perceptrons' },
      { id: 'activation-functions', label: 'Activation Functions' },
      { id: 'loss-functions', label: 'Loss Functions' },
      { id: 'backpropagation', label: 'Backpropagation Algorithm' }
    ]},
    { to: '/pytorch-nn-module', label: 'PyTorch nn.Module', subLinks: [
      { id: 'module-basics', label: 'Module Basics' },
      { id: 'custom-layers', label: 'Creating Custom Layers' },
      { id: 'parameter-management', label: 'Parameter Management' },
      { id: 'forward-method', label: 'Forward Method' },
      { id: 'model-composition', label: 'Model Composition' }
    ]},
    { to: '/training-neural-networks', label: 'Training Neural Networks', subLinks: [
      { id: 'training-loop', label: 'The Training Loop' },
      { id: 'optimizers', label: 'Optimizers' },
      { id: 'learning-rate-scheduling', label: 'Learning Rate Scheduling' },
      { id: 'regularization', label: 'Regularization Techniques' },
      { id: 'monitoring-training', label: 'Monitoring Training' }
    ]},
    { to: '/advanced-architectures', label: 'Advanced Architectures', subLinks: [
      { id: 'residual-networks', label: 'Residual Networks' },
      { id: 'attention-mechanisms', label: 'Attention Mechanisms' },
      { id: 'normalization', label: 'Normalization Techniques' },
      { id: 'skip-connections', label: 'Skip Connections' },
      { id: 'architectural-patterns', label: 'Architectural Patterns' }
    ]}
  ],
  'module3': [
    { to: '/convolutional-networks', label: 'Convolutional Neural Networks', subLinks: [
      { id: 'convolution-operation', label: 'Convolution Operation' },
      { id: 'cnn-architectures', label: 'CNN Architectures' },
      { id: 'pooling-layers', label: 'Pooling Layers' },
      { id: 'transfer-learning', label: 'Transfer Learning' },
      { id: 'computer-vision', label: 'Computer Vision Applications' }
    ]},
    { to: '/recurrent-networks', label: 'Recurrent Neural Networks', subLinks: [
      { id: 'rnn-basics', label: 'RNN Basics' },
      { id: 'lstm-gru', label: 'LSTM and GRU' },
      { id: 'sequence-to-sequence', label: 'Sequence-to-Sequence Models' },
      { id: 'attention-mechanism', label: 'Attention Mechanisms' },
      { id: 'nlp-applications', label: 'NLP Applications' }
    ]},
    { to: '/transformers', label: 'Transformer Architecture', subLinks: [
      { id: 'self-attention', label: 'Self-Attention' },
      { id: 'multi-head-attention', label: 'Multi-Head Attention' },
      { id: 'positional-encoding', label: 'Positional Encoding' },
      { id: 'transformer-blocks', label: 'Transformer Blocks' },
      { id: 'pre-trained-models', label: 'Pre-trained Models' }
    ]},
    { to: '/generative-models', label: 'Generative Models', subLinks: [
      { id: 'autoencoders', label: 'Autoencoders' },
      { id: 'variational-autoencoders', label: 'Variational Autoencoders' },
      { id: 'gans', label: 'Generative Adversarial Networks' },
      { id: 'diffusion-models', label: 'Diffusion Models' },
      { id: 'applications', label: 'Applications' }
    ]}
  ],
  'module4': [
    { to: '/model-optimization', label: 'Model Optimization', subLinks: [
      { id: 'quantization', label: 'Quantization' },
      { id: 'pruning', label: 'Model Pruning' },
      { id: 'knowledge-distillation', label: 'Knowledge Distillation' },
      { id: 'onnx', label: 'ONNX Export' },
      { id: 'torchscript', label: 'TorchScript' }
    ]},
    { to: '/deployment-strategies', label: 'Deployment Strategies', subLinks: [
      { id: 'serving-models', label: 'Model Serving' },
      { id: 'batch-inference', label: 'Batch Inference' },
      { id: 'real-time-inference', label: 'Real-time Inference' },
      { id: 'edge-deployment', label: 'Edge Deployment' },
      { id: 'cloud-deployment', label: 'Cloud Deployment' }
    ]},
    { to: '/monitoring-maintenance', label: 'Monitoring and Maintenance', subLinks: [
      { id: 'model-monitoring', label: 'Model Monitoring' },
      { id: 'performance-metrics', label: 'Performance Metrics' },
      { id: 'data-drift', label: 'Data Drift Detection' },
      { id: 'model-versioning', label: 'Model Versioning' },
      { id: 'continuous-integration', label: 'CI/CD for ML' }
    ]},
    { to: '/best-practices', label: 'Production Best Practices', subLinks: [
      { id: 'experiment-tracking', label: 'Experiment Tracking' },
      { id: 'reproducibility', label: 'Reproducibility' },
      { id: 'testing-ml', label: 'Testing ML Models' },
      { id: 'security', label: 'Security Considerations' },
      { id: 'ethics', label: 'Ethics and Fairness' }
    ]}
  ]
};

// PyTorch exercise content data
export const pytorchExerciseContentData = {
  'module1': [
    { to: '/exercise1', label: 'Exercise 1: Tensor Basics' },
    { to: '/exercise2', label: 'Exercise 2: PyTorch Operations' }
  ],
  'module2': [
    { to: '/exercise1', label: 'Exercise 1: Building Neural Networks' },
    { to: '/exercise2', label: 'Exercise 2: Training Networks' }
  ],
  'module3': [
    { to: '/exercise1', label: 'Exercise 1: Advanced Architectures' },
    { to: '/exercise2', label: 'Exercise 2: Specialized Networks' }
  ],
  'module4': [
    { to: '/exercise1', label: 'Exercise 1: Model Deployment' },
    { to: '/exercise2', label: 'Exercise 2: Production Monitoring' }
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
      { id: 'module1', name: 'PyTorch Core Components & Tensor Mathematics', icon: IconCube },
      { id: 'module2', name: 'Automatic Differentiation & Gradient Mechanics', icon: IconMathFunction },
      { id: 'module3', name: 'Infrastructure & Performance Optimization', icon: IconCpu },
      { id: 'module4', name: 'Advanced PyTorch Features & Custom Operations', icon: IconCode },
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