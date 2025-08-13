import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseAdvancedDL = () => {
  const courseLinks = [
    {
      to: "/mlp-architecture-components",
      label: "MLP Architecture & Components",
      component: lazy(() => import("./course/MLPArchitectureComponents")),
      subLinks: [
        { id: "multilayer-perceptron", label: "Multilayer Perceptron Mathematics" },
        { id: "universal-approximation", label: "Universal Approximation Theorem" },
        { id: "activation-functions", label: "Activation Functions: Mathematical Properties" },
        { id: "weight-initialization", label: "Weight Initialization Theory" },
        { id: "regularization-techniques", label: "Regularization Techniques (Dropout, L2, Batch Norm)" }
      ],
    },
    {
      to: "/data-pipeline-training-loop",
      label: "Data Pipeline & Training Loop",
      component: lazy(() => import("./course/DataPipelineTrainingLoop")),
      subLinks: [
        { id: "dataloader-architecture", label: "DataLoader Architecture & Multiprocessing" },
        { id: "batch-sampling", label: "Batch Sampling Strategies" },
        { id: "training-dynamics", label: "Training Dynamics & Loss Landscapes" },
        { id: "early-stopping", label: "Early Stopping & Convergence Criteria" }
      ],
    },
    {
      to: "/monitoring-visualization",
      label: "Monitoring & Visualization",
      component: lazy(() => import("./course/MonitoringVisualization")),
      subLinks: [
        { id: "tensorboard-integration", label: "TensorBoard Integration" },
        { id: "metrics-visualization", label: "Metrics Visualization Strategies" },
        { id: "model-interpretability", label: "Model Interpretability Basics" },
        { id: "debugging-networks", label: "Debugging Neural Networks" },
        { id: "checkpoint-saving", label: "Checkpoint Saving/Loading Strategies" }
      ],
    }
  ];

  const location = useLocation();
  const module = 3;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 3: Neural Networks & Training Infrastructure"
      courseLinks={courseLinks}
      enableSlides={true}
    >
      {location.pathname === `/courses/python-deep-learning/module${module}/course` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Build complete neural network training infrastructure with monitoring and visualization.</p>
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

export default CourseAdvancedDL;