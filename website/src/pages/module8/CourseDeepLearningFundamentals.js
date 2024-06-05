import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

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
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={8}
          isCourse={true}
          title="Module 8: Deep Learning Fundamentals"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about the fundamentals of deep learning
          and how to build and train neural networks using PyTorch.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module8/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseDeepLearningFundamentals;
