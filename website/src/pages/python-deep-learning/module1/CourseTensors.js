import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseTensors = () => {
  const courseLinks = [
    {
      to: "/deep-learning-introduction",
      label: "Deep Learning Introduction",
      component: lazy(() => import("./course/DeepLearningIntroduction")),
      subLinks: [
        { id: "what-is-deep-learning", label: "What is Deep Learning?" },
        { id: "pytorch-ecosystem", label: "PyTorch Ecosystem Overview" },
        { id: "comparison-frameworks", label: "Comparison with Other Frameworks" }
      ],
    },
    {
      to: "/mathematical-prerequisites",
      label: "Mathematical Prerequisites",
      component: lazy(() => import("./course/MathematicalPrerequisites")),
      subLinks: [
        { id: "model-parameters", label: "Model Parameters & Parameter Spaces" },
        { id: "loss-functions", label: "Loss Functions from Mathematical Perspective" },
        { id: "gradient-descent", label: "Gradient Descent Variants" },
        { id: "convergence-theory", label: "Convergence Theory Basics" }
      ],
    },
    {
      to: "/tensor-operations-computational-graphs",
      label: "Tensor Operations & Computational Graphs",
      component: lazy(() => import("./course/TensorOperationsComputationalGraphs")),
      subLinks: [
        { id: "tensor-algebra", label: "Tensor Algebra and Operations" },
        { id: "broadcasting", label: "Broadcasting Mechanics & Memory" },
        { id: "storage-views", label: "Storage, Views & Memory Layout" },
        { id: "computational-graphs", label: "Introduction to Computational Graphs" }
      ],
    }
  ];

  const location = useLocation();
  const module = 1;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 1: Foundations & Mathematical Framework"
      courseLinks={courseLinks}
      enableSlides={true}
    >
      {location.pathname === `/courses/python-deep-learning/module${module}/course` && (
        <>
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

export default CourseTensors;