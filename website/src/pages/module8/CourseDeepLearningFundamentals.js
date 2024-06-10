import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const CourseDeepLearningFundamentals = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Deep Learning",
      component: lazy(() => import("pages/module8/course/Introduction")),
    },
    {
      to: "/pytorch-basics",
      label: "PyTorch Basics",
      component: lazy(() => import("pages/module8/course/PyTorchBasics")),
    },
    {
      to: "/building-neural-networks",
      label: "Building Neural Networks",
      component: lazy(() =>
        import("pages/module8/course/BuildingNeuralNetworks")
      ),
    },
    {
      to: "/convolutional-neural-networks",
      label: "Convolutional Neural Networks (CNNs)",
      component: lazy(() =>
        import("pages/module8/course/ConvolutionalNeuralNetworks")
      ),
    },
    {
      to: "/recurrent-neural-networks-lstms",
      label: "Recurrent Neural Networks (RNNs) and LSTMs",
      component: lazy(() =>
        import("pages/module8/course/RecurrentNeuralNetworksLSTMs")
      ),
    },
    {
      to: "/training-deep-networks",
      label: "Training Deep Networks",
      component: lazy(() =>
        import("pages/module8/course/TrainingDeepNetworks")
      ),
    },
    {
      to: "/advanced-topics",
      label: "Advanced Topics in Deep Learning",
      component: lazy(() => import("pages/module8/course/AdvancedTopics")),
    },
  ];

  return (
    <ModuleFrame
      module={8}
      isCourse={true}
      title="Module 8: Deep Learning Fundamentals"
      courseLinks={courseLinks}
    >
      <Row>
        <p>
          In this module, you will learn about the fundamentals of deep learning
          and how to build and train neural networks using PyTorch.
        </p>
      </Row>
      <Row>
        <Col>
          <p>Last Updated: {"2024-06-07"}</p>
        </Col>
      </Row>
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseDeepLearningFundamentals;
