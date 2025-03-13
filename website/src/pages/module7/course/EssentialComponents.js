import React from 'react';
import { Title, Text, Stack, Accordion, Container } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';

// Import components
import Activation from './EssentialComponents/Activation';
import WeightInitialization from './EssentialComponents/WeightInitialization';
import Intro from './EssentialComponents/Intro';
import Optimization from './EssentialComponents/Optimization';
import TrainingBasics from './EssentialComponents/TrainingBasics';
import Dropout from './EssentialComponents/Dropout';
import EarlyStopping from './EssentialComponents/EarlyStopping';
import CustomLoss from './EssentialComponents/CustomLoss';
import CategoricalEmbeddings from './EssentialComponents/CategoricalEmbeddings';
import BatchNormalization from './EssentialComponents/BatchNormalization';
import ReduceLROnPlateau from './EssentialComponents/ReduceLROnPlateau';
import ResidualConnections from './EssentialComponents/ResidualConnections';

const EssentialComponents = () => {
  return (
    <Container fluid>
      <Stack spacing="xl">
        <Title id="common-issues" order={1}>Essential Components of Neural Networks</Title>

        <Intro />

        {/* Components Accordion */}
        <Accordion variant="separated">
          <Accordion.Item value="training-basics">
            <Accordion.Control>
              <Title order={3} id="basics">Basics</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <TrainingBasics />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="activation">
            <Accordion.Control>
              <Title order={3} id="activation">Activation Functions</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Activation />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="weight-init">
            <Accordion.Control>
              <Title order={3} id="weight-initialization">Weight Initialization</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <WeightInitialization />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="optimization">
            <Accordion.Control>
              <Title order={3} id="optimization">Optimization Techniques</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Optimization />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="dropout">
            <Accordion.Control>
              <Title order={3} id="dropout">Dropout</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Dropout />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="early-stopping">
            <Accordion.Control>
              <Title order={3} id="early-stopping">Early Stopping</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <EarlyStopping />
            </Accordion.Panel>
          </Accordion.Item>
          
          <Accordion.Item value="categorical-embeddings">
            <Accordion.Control>
              <Title order={3} id="categorical-embeddings">Categorical Variables & Embeddings</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <CategoricalEmbeddings />
            </Accordion.Panel>
          </Accordion.Item>
          
          <Accordion.Item value="custom-loss">
            <Accordion.Control>
              <Title order={3} id="custom-loss">Custom Loss Functions</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <CustomLoss />
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="batch-norm">
            <Accordion.Control>
              <Title order={3} id="batch-normalization">Batch Normalization</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <BatchNormalization />
            </Accordion.Panel>
          </Accordion.Item>
          
          <Accordion.Item value="reduce-lr">
            <Accordion.Control>
              <Title order={3} id="reduce-lr">Learning Rate Scheduling</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <ReduceLROnPlateau />
            </Accordion.Panel>
          </Accordion.Item>
          
          <Accordion.Item value="skip-connections">
            <Accordion.Control>
              <Title order={3} id="skip-connections">Skip Connections</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <ResidualConnections />
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      </Stack>
    </Container>
  );
};

export default EssentialComponents;