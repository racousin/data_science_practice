import React from 'react';
import { Title, Text, Stack, Accordion, Container } from '@mantine/core';
import 'katex/dist/katex.min.css';

// Import components
import ConvolutionBasics from './CNNEssentials/ConvolutionBasics';
import Pooling from './CNNEssentials/Pooling';
import Architectures from './CNNEssentials/Architectures';
import CNNBackpropagation from './CNNEssentials/CNNBackpropagation';

const CNNEssentials = () => {
  return (
    <Container size="lg">
      <Title id="cnn-essentials" order={1} mb="xl">Essential Components of Convolutional Neural Networks</Title>

      <Text size="lg" mb="xl">
        Convolutional Neural Networks (CNNs) are specialized deep learning architectures
        designed for processing grid-like data. The fundamental components of CNNs:
      </Text>

      <div data-slide>
        <Title order={2} id="convolution" mb="md">Convolution Operations</Title>
        <ConvolutionBasics />
      </div>

      <div data-slide>
        <Title order={2} id="pooling" mb="md">Pooling</Title>
        <Pooling />
      </div>

      <div data-slide>
        <Title order={2} id="architectures" mb="md">CNN Architectures</Title>
        <Architectures />
      </div>

      <Accordion variant="separated" mt="xl">
        <Accordion.Item value="cnn-backpropagation">
          <Accordion.Control>
            <Title order={2} id="cnn-backpropagation">CNN Backpropagation</Title>
          </Accordion.Control>
          <Accordion.Panel>
            <CNNBackpropagation />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Container>
  );
};

export default CNNEssentials;