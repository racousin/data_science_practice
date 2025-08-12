import React, { lazy } from 'react';
import { Container, Text, Stack } from '@mantine/core';
import DynamicRoutes from 'components/DynamicRoutes';
import ModuleFrame from 'components/ModuleFrame';
import { useLocation } from 'react-router-dom';

const CourseDeepLearningFundamentals = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction",
      component: lazy(() => import("pages/data-science-practice/module7/course/Introduction")),
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
          id: "artificial-neuron",
          label: "The Artificial Neuron"
        },
        {
          id: "network-structure",
          label: "Network Structure"
        },
        {
          id: "model-capacity",
          label: "Model Capacity and Depth"
        }
      ]
    },
    {
      to: "/backpropagation",
      label: "Backpropagation the learning Algorithm",
      component: lazy(() => import("pages/data-science-practice/module7/course/Backpropagation")),
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
      to: "/EssentialComponents",
      label: "Essential Components",
      component: lazy(() => import("pages/data-science-practice/module7/course/EssentialComponents")),
      subLinks: [
        {
          id: "common-issues",
          label: "Common Learning Issues"
        },
        {
          id: "basics",
          label: "Training Basics"
        },
        {
          id: "activation",
          label: "Activation Functions"
        },
        {
          id: "weight-initialization",
          label: "Weight Initialization"
        },
        {
          id: "optimization",
          label: "Optimization Techniques"
        },
        {
          id: "dropout",
          label: "Dropout"
        },
        {
          id: "early-stopping",
          label: "Early Stopping"
        },
        {
          id: "categorical-embeddings",
          label: "Categorical Variables & Embeddings"
        },
        {
          id: "custom-loss",
          label: "Custom Loss Functions"
        },
        {
          id: "batch-normalization",
          label: "Batch Normalization"
        },
        {
          id: "reduce-lr",
          label: "Learning Rate Scheduling"
        },
        {
          id: "residual-connections",
          label: "Residual Connections"
        }

      ]
    },
    {
      to: "/nn-workflow",
      label: "NNWorkflow",
      component: lazy(() => import("pages/data-science-practice/module7/course/NNWorkflow")),
      subLinks: [
        {
          id: "nn-workflow",
          label: "Neural Network Training Workflow"
        },
        {
          id: "example-data",
          label: "Example Regression Problem"
        },
        {
          id: "data-preparation",
          label: "Data Preparation"
        },
        {
          id: "device-setup",
          label: "Device Setup"
        },
        {
          id: "training-evaluation",
          label: "Training and Evaluation"
        },
        {
          id: "save-load-model",
          label: "Save and Load Model"
        },
        {
          id: "hyperparameter-optimization",
          label: "Hyperparameter Optimization"
        }
      ]
    },
    {
      to: "/case-study",
      label: "Case Study",
      component: lazy(() => import("pages/data-science-practice/module7/course/CaseStudy")),
      subLinks: [
      ]
    },

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

          <Text mt="md" c="dimmed" size="sm">
            Last Updated: 2024-09-20
          </Text>
        </Stack>
      )}
      <Container fluid p={0}>
        <DynamicRoutes routes={courseLinks} type="course" />
      </Container>
    </ModuleFrame>
  );
};

export default CourseDeepLearningFundamentals;