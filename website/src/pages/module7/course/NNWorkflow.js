import React from 'react';
import { Stack, Title, Text, Accordion } from '@mantine/core';
import { Database, Cpu, LineChart, Save, Settings } from 'lucide-react';
import CodeBlock from 'components/CodeBlock';

import DataPrep from './NNWorkflow/DataPrep';
import TrainEval from './NNWorkflow/TrainEval';
import Device from './NNWorkflow/Device';
import SaveLoad from './NNWorkflow/SaveLoad';
import HyperparameterOptimization from './NNWorkflow/HyperparameterOptimization';

const NNWorkflow = () => {
  return (
    <Stack spacing="xl">
      <Title order={1} id="nn-workflow">Neural Network Training Workflow</Title>
      
      <Text size="lg">
        Neural network training follows a structured workflow similar to classical machine learning, 
        with specific considerations for deep learning.
      </Text>

      <Title order={2} id="example-data">Example Regression Problem</Title>
      
      <Text>
        To illustrate the workflow, we'll use a simple nonlinear regression problem: 
        predicting y = 0.2x² + 0.5x + 2 with added Gaussian noise.
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch
import numpy as np

# Generate synthetic regression data
X = np.linspace(-5, 5, 1000).reshape(-1, 1)
y = 0.2 * X**2 + 0.5 * X + 2 + np.random.normal(0, 0.2, X.shape)`}
      />

      <Accordion variant="separated">
        <Accordion.Item value="data-prep">
          <Accordion.Control icon={<Database size={20} />}>
            <div id="data-preparation">Data Preparation</div>
          </Accordion.Control>
          <Accordion.Panel>
            <DataPrep />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="device">
          <Accordion.Control icon={<Cpu size={20} />}>
            <div id="device-setup">Device Setup</div>
          </Accordion.Control>
          <Accordion.Panel>
            <Device />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="train-eval">
          <Accordion.Control icon={<LineChart size={20} />}>
            <div id="training-evaluation">Training and Evaluation</div>
          </Accordion.Control>
          <Accordion.Panel>
            <TrainEval />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="save-load">
          <Accordion.Control icon={<Save size={20} />}>
            <div id="save-load-model">Save and Load Model</div>
          </Accordion.Control>
          <Accordion.Panel>
            <SaveLoad />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="hyperparameter">
          <Accordion.Control icon={<Settings size={20} />}>
            <div id="hyperparameter-optimization">Hyperparameter Optimization</div>
          </Accordion.Control>
          <Accordion.Panel>
            <HyperparameterOptimization />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Stack>
  );
};

export default NNWorkflow;