import React from 'react';
import { Title, Text, Stack, Accordion, Container } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';

// Import components
import ConvolutionBasics from './CNNEssentials/ConvolutionBasics';
import Pooling from './CNNEssentials/Pooling';
import Architectures from './CNNEssentials/Architectures';
import BackPropagation from './CNNEssentials/BackPropagation';
import Regularization from './CNNEssentials/Regularization';
import VisualizationTechniques from './CNNEssentials/VisualizationTechniques';
import PracticalConsiderations from './CNNEssentials/PracticalConsiderations';
import AdvancedConcepts from './CNNEssentials/AdvancedConcepts';

const CNNEssentials = () => {
  return (
    <Container fluid>
      <Stack spacing="xl">
        <Title id="cnn-essentials" order={1}>Essential Components of Convolutional Neural Networks</Title>
        
        <Text size="lg">
          Convolutional Neural Networks (CNNs) are specialized deep learning architectures 
          designed for processing grid-like data, particularly images. This module covers 
          the fundamental components and advanced concepts of CNNs with practical PyTorch implementations.
        </Text>

        <Accordion variant="separated">
          <Accordion.Item value="convolution-basics">
            <Accordion.Control>
              <Title order={3} id="convolution">Convolution Operations</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <ConvolutionBasics />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="pooling">
            <Accordion.Control>
              <Title order={3} id="pooling">Pooling and Downsampling</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Pooling />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="architectures">
            <Accordion.Control>
              <Title order={3} id="architectures">CNN Architectures</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Architectures />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="backprop">
            <Accordion.Control>
              <Title order={3} id="backprop">Backpropagation in CNNs</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <BackPropagation />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="regularization">
            <Accordion.Control>
              <Title order={3} id="regularization">CNN Regularization</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Regularization />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="visualization">
            <Accordion.Control>
              <Title order={3} id="visualization">Feature Visualization</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <VisualizationTechniques />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="practical">
            <Accordion.Control>
              <Title order={3} id="practical">Practical Considerations</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <PracticalConsiderations />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="advanced">
            <Accordion.Control>
              <Title order={3} id="advanced">Advanced Concepts</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <AdvancedConcepts />
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      </Stack>
    </Container>
  );
};

export default CNNEssentials;