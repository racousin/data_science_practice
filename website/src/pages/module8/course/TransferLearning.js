import React from 'react';
import { Container, Stack, Title, Text } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Book, GitBranch, Settings, Brain } from 'lucide-react';

import Fundamentals from './TransferLearning/Fundamentals';
import Implementation from './TransferLearning/Implementation';
import Optimization from './TransferLearning/Optimization';
import Applications from './TransferLearning/Applications';

const TransferLearning = () => {
  return (
    <Container size="lg">
      <Stack spacing="xl">
        <Title order={1} id="transfer-learning" mb="lg">
          Transfer Learning in Deep Learning
        </Title>
        
        <Text size="lg" mb="xl">
          Transfer learning is a machine learning technique where a model developed for one task
          is reused as the starting point for a model on a second task, significantly reducing
          training time and required data.
        </Text>

        <div id="fundamentals">
          <Title order={2} mb="md">
            <Book className="inline-block mr-2" size={24} />
            Fundamentals of Transfer Learning
          </Title>
          <Fundamentals />
        </div>

        <div id="implementation">
          <Title order={2} mb="md">
            <GitBranch className="inline-block mr-2" size={24} />
            Implementation Strategies
          </Title>
          <Implementation />
        </div>

        <div id="optimization">
          <Title order={2} mb="md">
            <Settings className="inline-block mr-2" size={24} />
            Fine-tuning and Optimization
          </Title>
          <Optimization />
        </div>

        <div id="applications">
          <Title order={2} mb="md">
            <Brain className="inline-block mr-2" size={24} />
            Practical Applications
          </Title>
          <Applications />
        </div>
      </Stack>
    </Container>
  );
};

export default TransferLearning;