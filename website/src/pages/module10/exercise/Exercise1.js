import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Network } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise1 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module10/exercise/module10_exercise1.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module10/exercise/module10_exercise1.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module10/exercise/module10_exercise1.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3"> 
            <Network size={24} />
            <Title order={1} className="text-2xl font-bold">Exercise 1: Transformer Warmup</Title>
          </div>

          <Stack spacing="lg">
            {/* Part 1 */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part 1: Implementing a Simple Transformer Model</Title>
              <Text className="text-gray-700 mb-4">
                Build and understand the core components of a Transformer architecture:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Self-attention mechanisms and multi-head attention</List.Item>
                <List.Item>Positional encodings and embedding layers</List.Item>
                <List.Item>Feed-forward networks and layer normalization</List.Item>
                <List.Item>Encoder-decoder architecture implementation</List.Item>
              </List>
            </div>

            {/* Part 2 */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part 2: Simple NLP Use Case - English to German Translation</Title>
              <Text className="text-gray-700 mb-4">
                Apply your transformer model to machine translation:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Preprocessing bilingual text data for translation</List.Item>
                <List.Item>Training a translation model with parallel corpora</List.Item>
                <List.Item>Implementing beam search for sequence generation</List.Item>
                <List.Item>Evaluating translation quality with BLEU scores</List.Item>
              </List>
            </div>
          </Stack>
        </Stack>
        <DataInteractionPanel
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          className="mt-6"
        />
      </Container>
    </>
  );
};

export default Exercise1;