import React, { lazy } from 'react';
import { Box, Container, Image } from '@mantine/core';
import { useLocation } from 'react-router-dom';
import DynamicRoutes from 'components/DynamicRoutes';
import ModuleFrame from 'components/ModuleFrame';

const ImageProcessingCourse = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction",
      component: lazy(() => import("pages/data-science-practice/module8/course/Introduction")),
      subLinks: [
        { id: "introduction", label: "Understanding Images in Deep Learning" },
        { id: "digital-representation", label: "Digital Image Representation" },
        { id: "ml-tasks", label: "Machine Learning Tasks with Images" },
        { id: "convolutions", label: "Understanding Convolutions" },
        {id: "cnn-architecture", label:"Convolutional Neural Networks"},
        { id: "efficiency", label: "Convolutional Layer Efficiency" }
      ]
    },
    {
      to: "/cnn-essentials",
      label: "Essential Components of CNNs",
      component: lazy(() => import("pages/data-science-practice/module8/course/CNNEssentials")),
      subLinks: [
        { id: "convolution", label: "Convolution Operations" },
        { id: "pooling", label: "Pooling" },
        { id: "architectures", label: "Popular CNN Architectures" },
        { id: "cnn-backpropagation", label: "CNN Backpropagation" }
      ]
    },
    {
      to: "/transfer-learning",
      label: "Transfer Learning",
      component: lazy(() => import("pages/data-science-practice/module8/course/TransferLearning")),
      subLinks: [
        { id: "transfer-learning", label: "Transfer Learning Basics" },
        { id: "hyperparameters", label: "Hyperparameter Optimization" },
        { id: "techniques", label: "Fine-tuning Techniques" }
      ]
    },
    {
      to: "/enhancement",
      label: "Image Enhancement Techniques",
      component: lazy(() => import("pages/data-science-practice/module8/course/Enhancement")),
      subLinks: [
        { id: "filtering", label: "Image Filtering" },
        { id: "histogram", label: "Histogram Equalization" },
        { id: "noise-reduction", label: "Noise Reduction Methods" }
      ]
    },
    // {
    //   to: "/object-detection",
    //   label: "Object Detection",
    //   component: lazy(() => import("pages/data-science-practice/module8/course/ObjectDetection")),
    //   subLinks: [
    //     { id: "rcnn-family", label: "R-CNN Family" },
    //     { id: "yolo", label: "YOLO Architecture" },
    //     { id: "metrics", label: "Evaluation Metrics" }
    //   ]
    // },
    // {
    //   to: "/segmentation",
    //   label: "Segmentation",
    //   component: lazy(() => import("pages/data-science-practice/module8/course/Segmentation")),
    //   subLinks: [
    //     { id: "semantic", label: "Semantic Segmentation" },
    //     { id: "instance", label: "Instance Segmentation" },
    //     { id: "evaluation", label: "Performance Evaluation" }
    //   ]
    // },
    {
      to: "/case-study",
      label: "Case Study",
      component: lazy(() => import("pages/data-science-practice/module8/course/CaseStudy")),
      subLinks: [
        { id: "problem", label: "Problem Definition" },
        { id: "implementation", label: "Implementation Steps" },
        { id: "results", label: "Results Analysis" }
      ]
    }
  ];return (
    <ModuleFrame
      module={8}
      isCourse={true}
      title="Image Processing"
      courseLinks={courseLinks}
    >
      <Container size="xl" px="md">
        <Box>
          <DynamicRoutes routes={courseLinks} type="course" />
        </Box>
      </Container>
    </ModuleFrame>
  );
};

export default ImageProcessingCourse;