import React from 'react';
import { Title, Text, Stack, Accordion, Container, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';

// Import components
import ConvolutionBasics from './CNNEssentials/ConvolutionBasics';
import Pooling from './CNNEssentials/Pooling';
import Architectures from './CNNEssentials/Architectures';
import CNNBackpropagation from './CNNEssentials/CNNBackpropagation';

const CNNEssentials = () => {
  return (
    <Container fluid>
      <Stack spacing="xl">
        <Title id="cnn-essentials" order={1}>Essential Components of Convolutional Neural Networks</Title>
        
        <Text size="lg">
          Convolutional Neural Networks (CNNs) are specialized deep learning architectures 
          designed for processing grid-like data. The fundamental components of CNNs:
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
              <Title order={3} id="pooling">Pooling</Title>
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
          <Accordion.Item value="cnn-backpropagation">
            <Accordion.Control>
              <Title order={3} id="cnn-backpropagation">CNN Backpropagation</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <CNNBackpropagation />
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      </Stack>
    </Container>
  );
};

export default CNNEssentials;