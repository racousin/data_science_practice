import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseNeuralNetworks = () => {
  const courseLinks = [
    {
      to: "/autograd-deep-dive",
      label: "Autograd Deep Dive",
      component: lazy(() => import("./course/AutogradDeepDive")),
      subLinks: [
        { id: "forward-reverse-mode", label: "Forward & Reverse Mode Differentiation" },
        { id: "computational-graph-construction", label: "Computational Graph Construction" },
        { id: "chain-rule-backpropagation", label: "Chain Rule & Backpropagation Mathematics" },
        { id: "gradient-accumulation", label: "Gradient Accumulation & Zeroing" }
      ],
    },
    {
      to: "/advanced-gradient-mechanics",
      label: "Advanced Gradient Mechanics",
      component: lazy(() => import("./course/AdvancedGradientMechanics")),
      subLinks: [
        { id: "gradient-flow", label: "Gradient Flow & Vanishing/Exploding" },
        { id: "gradient-clipping", label: "Gradient Clipping & Normalization" },
        { id: "higher-order-derivatives", label: "Higher-order Derivatives & Hessians" },
        { id: "custom-backward-passes", label: "Custom Backward Passes" }
      ],
    },
    {
      to: "/optimization-algorithms",
      label: "Optimization Algorithms",
      component: lazy(() => import("./course/OptimizationAlgorithms")),
      subLinks: [
        { id: "modern-optimizers", label: "Mathematical Foundations of Modern Optimizers" },
        { id: "adam-rmsprop-adagrad", label: "Adam, RMSprop, AdaGrad Derivations" },
        { id: "learning-rate-scheduling", label: "Learning Rate Scheduling Strategies" },
        { id: "second-order-optimization", label: "Second-order Optimization Methods" }
      ],
    }
  ];

  const location = useLocation();
  const module = 2;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 2: Automatic Differentiation & Optimization"
      courseLinks={courseLinks}
      enableSlides={true}
    >
      {location.pathname === `/courses/python-deep-learning/module${module}/course` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Master automatic differentiation and optimization algorithms for deep learning.</p>
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