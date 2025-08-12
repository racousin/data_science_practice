import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseNeuralNetworks = () => {
  const courseLinks = [
    {
      to: "/neural-network-basics",
      label: "Neural Network Basics",
      component: lazy(() => import("./course/NeuralNetworkBasics")),
      subLinks: [
        { id: "perceptron", label: "The Perceptron" },
        { id: "multilayer", label: "Multilayer Perceptrons" },
        { id: "activation-functions", label: "Activation Functions" },
        { id: "loss-functions", label: "Loss Functions" },
        { id: "backpropagation", label: "Backpropagation Algorithm" }
      ],
    },
    {
      to: "/pytorch-nn-module",
      label: "PyTorch nn.Module",
      component: lazy(() => import("./course/PyTorchNNModule")),
      subLinks: [
        { id: "module-basics", label: "Module Basics" },
        { id: "custom-layers", label: "Creating Custom Layers" },
        { id: "parameter-management", label: "Parameter Management" },
        { id: "forward-method", label: "Forward Method" },
        { id: "model-composition", label: "Model Composition" }
      ],
    },
    {
      to: "/training-neural-networks",
      label: "Training Neural Networks",
      component: lazy(() => import("./course/TrainingNeuralNetworks")),
      subLinks: [
        { id: "training-loop", label: "The Training Loop" },
        { id: "optimizers", label: "Optimizers" },
        { id: "learning-rate-scheduling", label: "Learning Rate Scheduling" },
        { id: "regularization", label: "Regularization Techniques" },
        { id: "monitoring-training", label: "Monitoring Training" }
      ],
    },
    {
      to: "/advanced-architectures",
      label: "Advanced Architectures",
      component: lazy(() => import("./course/AdvancedArchitectures")),
      subLinks: [
        { id: "residual-networks", label: "Residual Networks" },
        { id: "attention-mechanisms", label: "Attention Mechanisms" },
        { id: "normalization", label: "Normalization Techniques" },
        { id: "skip-connections", label: "Skip Connections" },
        { id: "architectural-patterns", label: "Architectural Patterns" }
      ],
    }
  ];

  const location = useLocation();
  const module = 2;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 2: Neural Network Architectures"
      courseLinks={courseLinks}
      enableSlides={true}
    >
      {location.pathname === `/courses/python-deep-learning/module${module}/course` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Master neural network architectures and PyTorch's nn.Module system.</p>
            </Grid.Col>
          </Grid>
          <Grid>
            <Grid.Col>
              <p>Last Updated: {"2025-01-12"}</p>
            </Grid.Col>
          </Grid>
        </>
      )}
      <Grid>
        <Grid.Col span={{ md: 11 }}>
          <DynamicRoutes routes={courseLinks} type="course" />
        </Grid.Col>
      </Grid>
    </ModuleFrame>
  );
};

export default CourseNeuralNetworks;