import React, { lazy } from 'react';
import { Container, Stack, Text, Title } from '@mantine/core';
import DynamicRoutes from 'components/DynamicRoutes';
import ModuleFrame from 'components/ModuleFrame';
import { useLocation } from 'react-router-dom';

const CourseDeepLearningFundamentals = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Deep Learning",
      component: lazy(() => import("pages/module7/course/Introduction")),
      subLinks: [
        {
          id: "historical-context",
          label: "Historical Context and Evolution"
        },
        {
          id: "frameworks",
          label: "Deep Learning Frameworks"
        },
        {
          id: "backpropagation",
          label: "Backpropagation and AutoDiff"
        },
        {
          id: "basic-example",
          label: "Simple Neural Network Example"
        }
      ]
    },
    {
      to: "/architecture",
      label: "Neural Network Architecture",
      component: lazy(() => import("pages/module7/course/Architecture")),
      subLinks: [
        {
          id: "network-structure",
          label: "Network Structure and Components"
        },
        {
          id: "layers",
          label: "Understanding Layers"
        },
        {
          id: "model-capacity",
          label: "Model Capacity and Depth"
        },
        {
          id: "computations",
          label: "Forward and Backward Computations"
        }
      ]
    },
    {
      to: "/activation",
      label: "Activation Functions",
      component: lazy(() => import("pages/module7/course/Activation")),
      subLinks: [
        {
          id: "purpose",
          label: "Role of Activation Functions"
        },
        {
          id: "common-functions",
          label: "Common Activation Functions"
        },
        {
          id: "properties",
          label: "Mathematical Properties"
        },
        {
          id: "usage-guidelines",
          label: "Usage Guidelines"
        }
      ]
    },
    {
      to: "/optimization",
      label: "Optimization Techniques",
      component: lazy(() => import("pages/module7/course/Optimization")),
      subLinks: [
        {
          id: "loss-functions",
          label: "Loss Functions"
        },
        {
          id: "optimizers",
          label: "Common Optimizers"
        },
        {
          id: "math-formulations",
          label: "Mathematical Foundations"
        },
        {
          id: "hyperparameters",
          label: "Hyperparameter Impact"
        }
      ]
    },
    {
      to: "/initialization",
      label: "Weight Initialization",
      component: lazy(() => import("pages/module7/course/Initialization")),
      subLinks: [
        {
          id: "importance",
          label: "Importance of Initialization"
        },
        {
          id: "methods",
          label: "Initialization Methods"
        },
        {
          id: "guidelines",
          label: "Selection Guidelines"
        }
      ]
    },
    {
      to: "/regularization",
      label: "Regularization Methods",
      component: lazy(() => import("pages/module7/course/Regularization")),
      subLinks: [
        {
          id: "overfitting",
          label: "Understanding Overfitting"
        },
        {
          id: "techniques",
          label: "Regularization Techniques"
        },
        {
          id: "implementation",
          label: "PyTorch Implementation"
        }
      ]
    },
    {
      to: "/advanced",
      label: "Advanced Topics",
      component: lazy(() => import("pages/module7/course/Advanced")),
      subLinks: [
        {
          id: "hyperparameter-opt",
          label: "Hyperparameter Optimization"
        },
        {
          id: "custom-loss",
          label: "Custom Loss Functions"
        },
        {
          id: "advanced-reg",
          label: "Advanced Regularization"
        },
        {
          id: "batch-norm",
          label: "Batch Normalization"
        }
      ]
    },
    {
      to: "/case-study",
      label: "Case Study",
      component: lazy(() => import("pages/module7/course/CaseStudy")),
      subLinks: [
        {
          id: "dataset",
          label: "Dataset Analysis"
        },
        {
          id: "model-building",
          label: "Model Architecture"
        },
        {
          id: "training",
          label: "Training Process"
        },
        {
          id: "evaluation",
          label: "Results and Visualization"
        }
      ]
    }
  ];

  const location = useLocation();
  const module = 7;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 7: Deep Learning Fundamentals"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <Stack spacing="md">
          <Text>
            This comprehensive module covers the fundamentals of deep learning, from basic neural network 
            concepts to advanced architectures and training techniques. The course is designed for students 
            with a strong mathematical background and focuses on both theoretical understanding and 
            practical implementation using PyTorch 2.5.
          </Text>
          
          <Title order={2} mt="md">Course Overview</Title>
          <Text>
            Throughout this module, you will learn:
          </Text>
          <ul>
            <li>Fundamental concepts of neural networks and deep learning</li>
            <li>Mathematical foundations of backpropagation and optimization</li>
            <li>Implementation of various neural network architectures</li>
            <li>Best practices for training and optimizing deep learning models</li>
            <li>Advanced techniques for improving model performance</li>
          </ul>

          <Text mt="sm">
            Each section includes theoretical explanations, mathematical formulations, 
            practical code examples, and interactive demonstrations using PyTorch 2.5.
          </Text>

          <Text mt="md" c="dimmed" size="sm">
            Last Updated: 2024-09-20
          </Text>
        </Stack>
      )}
      <Container fluid p={0}>
        <DynamicRoutes routes={courseLinks} />
      </Container>
    </ModuleFrame>
  );
};

export default CourseDeepLearningFundamentals;