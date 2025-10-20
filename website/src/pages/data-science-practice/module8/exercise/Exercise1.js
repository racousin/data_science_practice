import React from 'react';
import { Container, Text, Title, Stack, List, Alert, Badge } from '@mantine/core';
import { Brain } from 'lucide-react';
import { IconAlertCircle, IconStar } from '@tabler/icons-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise1 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise1.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise1.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/exercise/module8_exercise1.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Brain size={32} className="text-blue-600" />
            <Title order={1}>Exercise 1: Building a Small GPT-like Transformer</Title>
            <Badge color="yellow" size="lg" leftSection={<IconStar size={14} />}>
              Marked Exercise
            </Badge>
          </div>

          <Text size="md" mb="md">
            Build a decoder-only transformer architecture from scratch and train it as a language model.
            You will use the BPE tokenizer from Exercise 0 to prepare data and train a GPT-style model on TinyStories.
          </Text>

          <Stack spacing="lg">
            <div>
              <Title order={2} mb="md">Objectives</Title>
              <List spacing="sm">
                <List.Item>Build a GPT-style transformer architecture from scratch</List.Item>
                <List.Item>Implement self-attention and multi-head attention mechanisms</List.Item>
                <List.Item>Prepare training data using the BPE tokenizer from Exercise 0</List.Item>
                <List.Item>Train a language model for next-token prediction</List.Item>
                <List.Item>Generate text using different decoding strategies</List.Item>
                <List.Item>Evaluate model performance with perplexity metrics</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Dataset</Title>
              <Text size="md" mb="sm">
                <strong>TinyStories</strong> (<code>roneneldan/TinyStories</code>)
              </Text>
              <List spacing="sm">
                <List.Item>Same dataset as Exercise 0 for consistency</List.Item>
                <List.Item>Use the BPE tokenizer trained in Exercise 0</List.Item>
                <List.Item>Simple stories ideal for training small language models</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Key Topics</Title>
              <List spacing="sm">
                <List.Item>Decoder-only transformer architecture (GPT-style)</List.Item>
                <List.Item>Self-attention mechanisms and causal masking</List.Item>
                <List.Item>Multi-head attention implementation</List.Item>
                <List.Item>Positional encodings (learned or sinusoidal)</List.Item>
                <List.Item>Feed-forward networks and layer normalization</List.Item>
                <List.Item>Next-token prediction training objective</List.Item>
                <List.Item>Data batching and efficient training loops</List.Item>
                <List.Item>Text generation strategies: greedy, sampling, top-k, nucleus sampling</List.Item>
                <List.Item>Perplexity and loss evaluation</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Deliverables</Title>
              <List spacing="sm">
                <List.Item>Complete GPT-like transformer implementation in PyTorch</List.Item>
                <List.Item>Data preprocessing pipeline using Exercise 0 tokenizer</List.Item>
                <List.Item>Trained language model on TinyStories</List.Item>
                <List.Item>Text generation examples with different decoding strategies</List.Item>
                <List.Item>Training loss curves and perplexity analysis</List.Item>
                <List.Item>Generated story samples and quality assessment</List.Item>
              </List>
            </div>

            <Alert icon={<IconAlertCircle />} color="yellow" mt="md">
              This is a marked exercise. Make sure your implementation is well-documented and includes all required components.
              Pay special attention to the attention mechanism implementation and text generation quality.
            </Alert>
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

export default Exercise1;
