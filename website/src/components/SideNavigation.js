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
    { to: '/Introduction', label: 'Automatic Differentiation Fundamentals', subLinks: [
      { id: 'autograd-concept', label: 'Autograd Concept' },
      { id: 'computational-graphs', label: 'Computational Graphs' },
      { id: 'dynamic-vs-static', label: 'Dynamic vs Static Graphs' }
    ]},
    { to: '/gradient-computation', label: 'Gradient Computation', subLinks: [
      { id: 'backward-pass', label: 'The Backward Pass' },
      { id: 'chain-rule', label: 'Chain Rule Implementation' },
      { id: 'gradient-accumulation', label: 'Gradient Accumulation' },
      { id: 'gradient-clipping', label: 'Gradient Clipping' }
    ]},
    { to: '/autograd-mechanics', label: 'Autograd Mechanics', subLinks: [
      { id: 'requires-grad', label: 'requires_grad Mechanism' },
      { id: 'grad-fn', label: 'grad_fn & Function Objects' },
      { id: 'retain-graph', label: 'retain_graph & Multiple Backwards' },
      { id: 'no-grad-context', label: 'no_grad & inference_mode' }
    ]},
    { to: '/custom-autograd', label: 'Custom Autograd Functions', subLinks: [
      { id: 'function-class', label: 'torch.autograd.Function' },
      { id: 'custom-backward', label: 'Implementing Custom Backward' },
      { id: 'numerical-gradients', label: 'Numerical Gradient Checking' },
      { id: 'higher-order-derivatives', label: 'Higher-order Derivatives' }
    ]},
    { to: '/gradient-hooks', label: 'Gradient Hooks & Debugging', subLinks: [
      { id: 'tensor-hooks', label: 'Tensor Hooks' },
      { id: 'module-hooks', label: 'Module Hooks' },
      { id: 'gradient-debugging', label: 'Gradient Debugging Techniques' },
      { id: 'anomaly-detection', label: 'Anomaly Detection' }
    ]},
    { to: '/advanced-differentiation', label: 'Advanced Differentiation', subLinks: [
      { id: 'jacobian-hessian', label: 'Jacobian & Hessian Computation' },
      { id: 'functorch', label: 'Functorch Integration' },
      { id: 'double-backward', label: 'Double Backward Pass' },
      { id: 'gradient-checkpointing', label: 'Gradient Checkpointing' }
    ]}
  ],
  'module3': [
    { to: '/Introduction', label: 'Compute Infrastructure', subLinks: [
      { id: 'cpu-vs-gpu', label: 'CPU vs GPU Computation' },
      { id: 'device-management', label: 'Device Management' },
      { id: 'cuda-basics', label: 'CUDA Basics in PyTorch' }
    ]},
    { to: '/memory-optimization', label: 'Memory Optimization', subLinks: [
      { id: 'memory-profiling', label: 'Memory Profiling Tools' },
      { id: 'memory-efficient-attention', label: 'Memory-efficient Attention' },
      { id: 'activation-checkpointing', label: 'Activation Checkpointing' },
      { id: 'gradient-accumulation-memory', label: 'Memory-aware Gradient Accumulation' }
    ]},
    { to: '/performance-profiling', label: 'Performance Profiling', subLinks: [
      { id: 'torch-profiler', label: 'PyTorch Profiler' },
      { id: 'tensorboard-profiling', label: 'TensorBoard Profiling' },
      { id: 'bottleneck-identification', label: 'Bottleneck Identification' },
      { id: 'cuda-events', label: 'CUDA Events & Timing' }
    ]},
    { to: '/parallelization', label: 'Parallelization Strategies', subLinks: [
      { id: 'data-parallelism', label: 'Data Parallelism' },
      { id: 'model-parallelism', label: 'Model Parallelism' },
      { id: 'pipeline-parallelism', label: 'Pipeline Parallelism' },
      { id: 'threading-multiprocessing', label: 'Threading vs Multiprocessing' }
    ]},
    { to: '/compilation-optimization', label: 'Compilation & Optimization', subLinks: [
      { id: 'torchscript', label: 'TorchScript Compilation' },
      { id: 'torch-compile', label: 'torch.compile' },
      { id: 'onnx-export', label: 'ONNX Export & Optimization' },
      { id: 'quantization', label: 'Quantization Techniques' }
    ]},
    { to: '/distributed-basics', label: 'Distributed Computing Basics', subLinks: [
      { id: 'distributed-data-parallel', label: 'DistributedDataParallel' },
      { id: 'communication-backends', label: 'Communication Backends' },
      { id: 'synchronization', label: 'Synchronization Primitives' },
      { id: 'fault-tolerance', label: 'Fault Tolerance' }
    ]}
  ],
  'module4': [
    { to: '/Introduction', label: 'Custom Operations', subLinks: [
      { id: 'extending-pytorch', label: 'Extending PyTorch' },
      { id: 'cpp-extensions', label: 'C++ Extensions Overview' },
      { id: 'cuda-extensions', label: 'CUDA Extensions Overview' }
    ]},
    { to: '/custom-functions', label: 'Custom Autograd Functions', subLinks: [
      { id: 'advanced-function-class', label: 'Advanced Function Class Usage' },
      { id: 'save-for-backward', label: 'save_for_backward & ctx Usage' },
      { id: 'non-differentiable-inputs', label: 'Non-differentiable Inputs' },
      { id: 'inplace-functions', label: 'In-place Custom Functions' }
    ]},
    { to: '/cpp-extensions', label: 'C++ Extensions Deep Dive', subLinks: [
      { id: 'pybind11-basics', label: 'PyBind11 Basics' },
      { id: 'tensor-cpp-api', label: 'Tensor C++ API' },
      { id: 'autograd-cpp', label: 'Autograd in C++' },
      { id: 'building-extensions', label: 'Building & Installing Extensions' }
    ]},
    { to: '/cuda-programming', label: 'CUDA Programming', subLinks: [
      { id: 'cuda-kernels', label: 'Writing CUDA Kernels' },
      { id: 'memory-management-cuda', label: 'CUDA Memory Management' },
      { id: 'cooperative-groups', label: 'Cooperative Groups' },
      { id: 'cuda-graphs', label: 'CUDA Graphs' }
    ]},
    { to: '/debugging-profiling-advanced', label: 'Advanced Debugging & Profiling', subLinks: [
      { id: 'cuda-debugging', label: 'CUDA Debugging Tools' },
      { id: 'nsight-profiling', label: 'Nsight Profiling' },
      { id: 'memory-debugging', label: 'Memory Debugging' },
      { id: 'performance-regression', label: 'Performance Regression Testing' }
    ]},
    { to: '/production-deployment', label: 'Production & Deployment', subLinks: [
      { id: 'torchserve', label: 'TorchServe' },
      { id: 'model-optimization', label: 'Model Optimization for Production' },
      { id: 'monitoring-logging', label: 'Monitoring & Logging' },
      { id: 'a-b-testing', label: 'A/B Testing ML Models' }
    ]}
  ]
};

// PyTorch exercise content data
export const pytorchExerciseContentData = {
  'module1': [
    { to: '/exercise1', label: 'Tensor Creation & Manipulation' },
    { to: '/exercise2', label: 'Mathematical Operations Deep Dive' },
    { to: '/exercise3', label: 'Memory Management Challenges' }
  ],
  'module2': [
    { to: '/exercise1', label: 'Custom Autograd Functions' },
    { to: '/exercise2', label: 'Gradient Debugging Workshop' },
    { to: '/exercise3', label: 'Higher-order Derivatives' }
  ],
  'module3': [
    { to: '/exercise1', label: 'Performance Profiling Lab' },
    { to: '/exercise2', label: 'Memory Optimization Challenge' },
    { to: '/exercise3', label: 'Parallelization Patterns' }
  ],
  'module4': [
    { to: '/exercise1', label: 'Custom C++ Extension' },
    { to: '/exercise2', label: 'CUDA Kernel Implementation' },
    { to: '/exercise3', label: 'Production Optimization' }
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