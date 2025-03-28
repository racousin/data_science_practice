import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Image } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise0 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module8/exercise/module8_exercise0.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module8/exercise/module8_exercise0.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module8/exercise/module8_exercise0.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Image size={32} className="text-blue-600" />
            <Title order={1} className="text-2xl font-bold">Exercise 0: Image Processing & CNN Warmup</Title>
          </div>

          <Stack spacing="lg">
            {/* Part A */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part A: Image Processing Warm-up</Title>
              <Text className="text-gray-700 mb-4">
                Learn fundamental image processing operations using Python and OpenCV:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Loading and displaying images using OpenCV and Matplotlib</List.Item>
                <List.Item>Basic image manipulations (resize, crop, rotate)</List.Item>
                <List.Item>Color space conversions (RGB, BGR, Grayscale)</List.Item>
                <List.Item>Image normalization and preprocessing techniques</List.Item>
              </List>
            </div>

            {/* Part B */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part B: Convolution Warm-up</Title>
              <Text className="text-gray-700 mb-4">
                Implement and understand common convolution kernels:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Edge detection kernels (Sobel, Prewitt)</List.Item>
                <List.Item>Blurring kernels (Average, Gaussian)</List.Item>
                <List.Item>Sharpening kernels</List.Item>
                <List.Item>Manual convolution implementation</List.Item>
              </List>
            </div>

            {/* Part C */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part C: Torch CNN Warm-up</Title>
              <Text className="text-gray-700 mb-4">
                Introduction to CNN components using PyTorch:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Understanding convolutional layers and their parameters</List.Item>
                <List.Item>Implementing different types of pooling layers</List.Item>
                <List.Item>Building a basic CNN architecture</List.Item>
                <List.Item>Visualizing feature maps and filters</List.Item>
              </List>
            </div>
          </Stack>

          <DataInteractionPanel
            notebookUrl={notebookUrl}
            notebookHtmlUrl={notebookHtmlUrl}
            notebookColabUrl={notebookColabUrl}
            className="mt-6"
          />
        </Stack>
      </Container>
    </>
  );
};

export default Exercise0;