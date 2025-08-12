import React, { lazy } from "react";
import { Container, Grid } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseAdvancedDL = () => {
  const courseLinks = [
    {
      to: "/convolutional-networks",
      label: "Convolutional Neural Networks",
      component: lazy(() => import("./course/ConvolutionalNetworks")),
      subLinks: [
        { id: "convolution-operation", label: "Convolution Operation" },
        { id: "cnn-architectures", label: "CNN Architectures" },
        { id: "pooling-layers", label: "Pooling Layers" },
        { id: "transfer-learning", label: "Transfer Learning" },
        { id: "computer-vision", label: "Computer Vision Applications" }
      ],
    },
    {
      to: "/recurrent-networks",
      label: "Recurrent Neural Networks",
      component: lazy(() => import("./course/RecurrentNetworks")),
      subLinks: [
        { id: "rnn-basics", label: "RNN Basics" },
        { id: "lstm-gru", label: "LSTM and GRU" },
        { id: "sequence-to-sequence", label: "Sequence-to-Sequence Models" },
        { id: "attention-mechanism", label: "Attention Mechanisms" },
        { id: "nlp-applications", label: "NLP Applications" }
      ],
    },
    {
      to: "/transformers",
      label: "Transformer Architecture",
      component: lazy(() => import("./course/Transformers")),
      subLinks: [
        { id: "self-attention", label: "Self-Attention" },
        { id: "multi-head-attention", label: "Multi-Head Attention" },
        { id: "positional-encoding", label: "Positional Encoding" },
        { id: "transformer-blocks", label: "Transformer Blocks" },
        { id: "pre-trained-models", label: "Pre-trained Models" }
      ],
    },
    {
      to: "/generative-models",
      label: "Generative Models",
      component: lazy(() => import("./course/GenerativeModels")),
      subLinks: [
        { id: "autoencoders", label: "Autoencoders" },
        { id: "variational-autoencoders", label: "Variational Autoencoders" },
        { id: "gans", label: "Generative Adversarial Networks" },
        { id: "diffusion-models", label: "Diffusion Models" },
        { id: "applications", label: "Applications" }
      ],
    }
  ];

  const location = useLocation();
  const module = 3;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 3: Advanced Deep Learning"
      courseLinks={courseLinks}
      enableSlides={true}
    >
      {location.pathname === `/courses/python-deep-learning/module${module}/course` && (
        <>
          <Grid>
            <Grid.Col>
              <p>Explore advanced deep learning architectures: CNNs, RNNs, Transformers, and Generative Models.</p>
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