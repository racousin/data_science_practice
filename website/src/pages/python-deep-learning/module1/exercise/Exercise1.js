import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Cpu } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise1 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module1/exercises/exercise1.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module1/exercises/exercise1.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/python-deep-learning/module1/exercises/exercise1.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Cpu size={32} className="text-blue-600" />
            <Title order={1} className="text-2xl font-bold">Exercise 1: Tensor Basics</Title>
          </div>

          <Stack spacing="lg">
            {/* Part 1 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 1: Tensor Creation</Title>
              <Text className="text-gray-700 mb-4">
                Master different methods for creating PyTorch tensors:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Creating tensors with specific values (zeros, ones, identity)</List.Item>
                <List.Item>Random tensor generation with different distributions</List.Item>
                <List.Item>Converting from Python lists and NumPy arrays</List.Item>
                <List.Item>Using torch.arange and torch.linspace</List.Item>
              </List>
            </div>

            {/* Part 2 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 2: Tensor Attributes</Title>
              <Text className="text-gray-700 mb-4">
                Explore tensor properties and metadata:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Understanding tensor shape, dtype, and device</List.Item>
                <List.Item>Getting tensor dimensions and element count</List.Item>
                <List.Item>Checking tensor properties (is_cuda, is_sparse)</List.Item>
                <List.Item>Memory layout and stride information</List.Item>
              </List>
            </div>

            {/* Part 3 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 3: Indexing and Slicing</Title>
              <Text className="text-gray-700 mb-4">
                Access and modify tensor elements efficiently:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Basic indexing with integers and slices</List.Item>
                <List.Item>Multi-dimensional indexing</List.Item>
                <List.Item>Advanced slicing techniques</List.Item>
                <List.Item>Modifying tensor values through indexing</List.Item>
              </List>
            </div>

            {/* Part 4 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 4: Tensor Reshaping</Title>
              <Text className="text-gray-700 mb-4">
                Manipulate tensor dimensions:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Reshaping tensors with view and reshape</List.Item>
                <List.Item>Flattening and unflattening operations</List.Item>
                <List.Item>Adding and removing dimensions (squeeze/unsqueeze)</List.Item>
                <List.Item>Permuting and transposing dimensions</List.Item>
              </List>
            </div>

            {/* Part 5 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 5: Data Types</Title>
              <Text className="text-gray-700 mb-4">
                Work with different tensor data types:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Float tensors (float32, float64, float16)</List.Item>
                <List.Item>Integer tensors (int32, int64, int8)</List.Item>
                <List.Item>Boolean tensors and conditional operations</List.Item>
                <List.Item>Type casting and precision considerations</List.Item>
              </List>
            </div>

            {/* Part 6 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 6: NumPy Interoperability</Title>
              <Text className="text-gray-700 mb-4">
                Convert between PyTorch tensors and NumPy arrays:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Converting NumPy arrays to PyTorch tensors</List.Item>
                <List.Item>Converting PyTorch tensors to NumPy arrays</List.Item>
                <List.Item>Understanding shared memory behavior</List.Item>
                <List.Item>Handling device transfers (CPU/GPU)</List.Item>
              </List>
            </div>
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