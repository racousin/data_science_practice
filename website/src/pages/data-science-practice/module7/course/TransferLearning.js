import React from 'react';
import { Container, Stack, Title, Text } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import DataInteractionPanel from 'components/DataInteractionPanel';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Book, GitBranch, Settings, Brain, Code } from 'lucide-react';

import Fundamentals from './TransferLearning/Fundamentals';
import FineTuning from './TransferLearning/FineTuning';

const TransferLearning = () => {
  // Notebook URLs
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/course/module7_course_finetuning.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/course/module7_course_finetuning.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module7/course/module7_course_finetuning.ipynb";

  const metadata = {
    description: "Complete transfer learning application on CIFAR-10, comparing training from scratch vs. using pretrained models.",
    source: "CIFAR-10 Dataset",
    target: "Image classification (10 classes)",
    listData: [
      { name: "SimpleCNN", description: "Baseline model trained from scratch" },
      { name: "ResNet18", description: "Pretrained model with frozen/unfrozen backbone" },
      { name: "EfficientNet", description: "Modern architecture via timm library" }
    ],
  };

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

        <div id="practical-application">
          <Title order={2} mb="md" mt="xl">
            <Code className="inline-block mr-2" size={24} />
            Practical Application: CIFAR-10 Transfer Learning
          </Title>
          <Text mb="md">
            This notebook demonstrates a complete transfer learning pipeline on CIFAR-10,
            comparing different approaches from training from scratch to using modern pretrained architectures.
          </Text>
          <DataInteractionPanel
            notebookUrl={notebookUrl}
            notebookHtmlUrl={notebookHtmlUrl}
            notebookColabUrl={notebookColabUrl}
            metadata={metadata}
          />
        </div>
      </Stack>
    </Container>
  );
};

export default TransferLearning;