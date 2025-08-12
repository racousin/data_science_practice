import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseProduction = () => {
  const courseLinks = [
    {
      to: "/model-optimization",
      label: "Model Optimization",
      component: lazy(() => import("./course/ModelOptimization")),
      subLinks: [
        { id: "quantization", label: "Quantization" },
        { id: "pruning", label: "Model Pruning" },
        { id: "knowledge-distillation", label: "Knowledge Distillation" },
        { id: "onnx", label: "ONNX Export" },
        { id: "torchscript", label: "TorchScript" }
      ],
    },
    {
      to: "/deployment-strategies",
      label: "Deployment Strategies",
      component: lazy(() => import("./course/DeploymentStrategies")),
      subLinks: [
        { id: "serving-models", label: "Model Serving" },
        { id: "batch-inference", label: "Batch Inference" },
        { id: "real-time-inference", label: "Real-time Inference" },
        { id: "edge-deployment", label: "Edge Deployment" },
        { id: "cloud-deployment", label: "Cloud Deployment" }
      ],
    },
    {
      to: "/monitoring-maintenance",
      label: "Monitoring and Maintenance",
      component: lazy(() => import("./course/MonitoringMaintenance")),
      subLinks: [
        { id: "model-monitoring", label: "Model Monitoring" },
        { id: "performance-metrics", label: "Performance Metrics" },
        { id: "data-drift", label: "Data Drift Detection" },
        { id: "model-versioning", label: "Model Versioning" },
        { id: "continuous-integration", label: "CI/CD for ML" }
      ],
    },
    {
      to: "/best-practices",
      label: "Production Best Practices",
      component: lazy(() => import("./course/BestPractices")),
      subLinks: [
        { id: "experiment-tracking", label: "Experiment Tracking" },
        { id: "reproducibility", label: "Reproducibility" },
        { id: "testing-ml", label: "Testing ML Models" },
        { id: "security", label: "Security Considerations" },
        { id: "ethics", label: "Ethics and Fairness" }
      ],
    }
  ];

  const location = useLocation();
  const module = 4;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 4: Production and Deployment"
      courseLinks={courseLinks}
      enableSlides={true}
    >
      {location.pathname === `/courses/python-deep-learning/module${module}/course` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Learn to deploy, optimize, and maintain deep learning models in production environments.</p>
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

export default CourseProduction;