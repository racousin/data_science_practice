import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseProduction = () => {
  const courseLinks = [
    {
      to: "/device-management-resources",
      label: "Device Management & Resources",
      component: lazy(() => import("./course/DeviceManagementResources")),
      subLinks: [
        { id: "gpu-architecture", label: "GPU Architecture for Deep Learning" },
        { id: "memory-management", label: "Memory Management Strategies" },
        { id: "flops-memory-calculation", label: "Calculate FLOPs & Memory Requirements" },
        { id: "mixed-precision", label: "Mixed Precision Training Mathematics" }
      ],
    },
    {
      to: "/model-optimization",
      label: "Model Optimization",
      component: lazy(() => import("./course/ModelOptimization")),
      subLinks: [
        { id: "model-compression", label: "Model Compression Techniques" },
        { id: "computational-complexity", label: "Computational Complexity Analysis" },
        { id: "torchscript-serialization", label: "TorchScript & Model Serialization" },
        { id: "jit-compilation", label: "JIT Compilation Basics" }
      ],
    },
    {
      to: "/advanced-pytorch-architecture",
      label: "Advanced PyTorch & Architecture Overview",
      component: lazy(() => import("./course/AdvancedPyTorchArchitecture")),
      subLinks: [
        { id: "hooks-applications", label: "Hooks & Their Applications" },
        { id: "dynamic-computation-graphs", label: "Dynamic Computation Graphs" },
        { id: "cnn-convolution-mathematics", label: "CNN Convolution Mathematics (Brief)" },
        { id: "attention-mechanism-mathematics", label: "Attention Mechanism Mathematics (Brief)" },
        { id: "custom-cpp-extensions", label: "Custom C++ Extensions Overview" }
      ],
    }
  ];

  const location = useLocation();
  const module = 4;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 4: Performance Optimization & Advanced Features"
      courseLinks={courseLinks}
      enableSlides={true}
    >
      {location.pathname === `/courses/python-deep-learning/module${module}/course` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Master performance optimization and advanced PyTorch features for production systems.</p>
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