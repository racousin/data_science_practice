import React from 'react';
import { Container, Text, Title, List, Stack } from '@mantine/core';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise2 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/module7_exercise2.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/module7_exercise2.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module7/exercise/module7_exercise2.ipynb";

  const metadata = {
    description: "A dataset of 64x64 cat face images for training generative models. The dataset contains thousands of cat face images suitable for VAE, GAN, or Diffusion model training.",
    source: "Kaggle Cats Faces Dataset",
    target: "Generate synthetic cat face images",
    listData: [
      { name: "images", description: "64x64 RGB images of cat faces" },
      { name: "task", description: "Train a generative model to create realistic cat faces", isTarget: true }
    ],
  };

  return (
    <Container fluid className="p-4">
      <Stack spacing="lg">
        <Title order={1}>Exercise 2: Cat Face Generation</Title>

        <Stack spacing="md">
          <Title order={2} id="overview">Overview</Title>
          <List>
            <List.Item>Download the cats faces dataset from Kaggle</List.Item>
            <List.Item>Prepare the data for generative model training</List.Item>
            <List.Item>Choose between VAE, GAN, or Diffusion model</List.Item>
            <List.Item>Train your model to generate synthetic cat faces</List.Item>
            <List.Item>Evaluate the quality of generated images</List.Item>
          </List>

          <Title order={2} id="dataset">Dataset</Title>
          <Text>
            Dataset: Cats Faces 64x64 for Generative Models
          </Text>
          <Text>
            Download from: <a href="https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models" target="_blank" rel="noopener noreferrer">Kaggle</a>
          </Text>

          <Title order={2} id="model-options">Model Options</Title>

          <Title order={3}>Option 1: Variational Autoencoder (VAE)</Title>
          <List>
            <List.Item>Encoder network maps images to latent space distribution</List.Item>
            <List.Item>Decoder network reconstructs images from latent vectors</List.Item>
            <List.Item>Loss combines reconstruction error and KL divergence</List.Item>
            <List.Item>Generate by sampling from learned latent distribution</List.Item>
          </List>

          <Title order={3}>Option 2: Generative Adversarial Network (GAN)</Title>
          <List>
            <List.Item>Generator creates images from random noise vectors</List.Item>
            <List.Item>Discriminator distinguishes real from generated images</List.Item>
            <List.Item>Adversarial training using minimax objective</List.Item>
            <List.Item>Generate by feeding noise to trained generator</List.Item>
          </List>

          <Title order={3}>Option 3: Diffusion Model</Title>
          <List>
            <List.Item>Forward process gradually adds noise to images</List.Item>
            <List.Item>U-Net architecture predicts noise at each timestep</List.Item>
            <List.Item>Train to reverse the diffusion process</List.Item>
            <List.Item>Generate by iterative denoising from pure noise</List.Item>
          </List>
        </Stack>

        <DataInteractionPanel
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          metadata={metadata}
        />
      </Stack>
    </Container>
  );
};

export default Exercise2;
