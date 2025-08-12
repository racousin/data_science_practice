import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseTensors = () => {
  const courseLinks = [
    {
      to: "/tensor-fundamentals",
      label: "Tensor Fundamentals",
      component: lazy(() => import("./course/TensorFundamentals")),
      subLinks: [
        { id: "dimensions", label: "Understanding Dimensions" },
        { id: "creation", label: "Creating Tensors" },
        { id: "operations", label: "Tensor Operations" },
        { id: "gpu", label: "GPU Acceleration" },
        { id: "autograd", label: "Automatic Differentiation" },
        { id: "broadcasting", label: "Broadcasting" }
      ],
    },
  ];

  const location = useLocation();
  const module = 1;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 1: Introduction to Tensors"
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