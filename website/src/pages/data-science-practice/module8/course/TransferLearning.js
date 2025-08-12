import React from 'react';
import { Container, Stack, Title, Text } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Book, GitBranch, Settings, Brain } from 'lucide-react';

import Fundamentals from './TransferLearning/Fundamentals';
import FineTuning from './TransferLearning/FineTuning';

const TransferLearning = () => {
  return (
    <Container size="lg">
      <Stack spacing="xl">
        <Title order={1} id="transfer-learning" mb="lg">
          Transfer Learning in Deep Learning
        </Title>
        
        <Text size="lg" mb="xl">
          Transfer learning enables leveraging pre-trained models to solve new tasks efficiently,
          reducing training time and data requirements while potentially improving performance.
        </Text>

        <div id="fundamentals">
          <Title order={2} mb="md">
            <Book className="inline-block mr-2" size={24} />
            Model Selection and Adaptation
          </Title>
          <Fundamentals />
        </div>

        <div id="fine-tuning">
          <Title order={2} mb="md" mt="xl">
            <Settings className="inline-block mr-2" size={24} />
            Fine-tuning Strategies
          </Title>
          <FineTuning />
        </div>
      </Stack>
    </Container>
  );
};

export default TransferLearning;