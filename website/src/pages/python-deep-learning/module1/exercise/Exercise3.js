import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Cpu } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise3 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module1/exercises/exercise3.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module1/exercises/exercise3.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/python-deep-learning/module1/exercises/exercise3.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Cpu size={32} className="text-blue-600" />
            <Title order={1} className="text-2xl font-bold">Exercise 1.3: Tensor Mastery</Title>
          </div>

          <Stack spacing="lg">
            {/* Part 1 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 1: Advanced Tensor Manipulations</Title>
              <Text className="text-gray-700 mb-4">
                Master complex tensor operations and indexing:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Fancy indexing with integer arrays and boolean masks</List.Item>
                <List.Item>Gather and scatter operations for advanced selection</List.Item>
                <List.Item>Tensor concatenation, stacking, and splitting</List.Item>
                <List.Item>Advanced reshaping with einops-style operations</List.Item>
              </List>
            </div>

            {/* Part 2 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 2: Memory Profiling Exercises</Title>
              <Text className="text-gray-700 mb-4">
                Analyze and optimize tensor memory usage:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Measuring memory footprint of different tensor operations</List.Item>
                <List.Item>Understanding views vs. copies in tensor operations</List.Item>
                <List.Item>Memory-efficient operations and in-place modifications</List.Item>
                <List.Item>Profiling memory allocation and deallocation patterns</List.Item>
              </List>
            </div>

            {/* Part 3 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 3: Matrix Operations Benchmarking</Title>
              <Text className="text-gray-700 mb-4">
                Compare performance of different matrix operation approaches:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Element-wise vs. vectorized operations performance</List.Item>
                <List.Item>CPU vs. GPU performance for different matrix sizes</List.Item>
                <List.Item>Memory-bound vs. compute-bound operation analysis</List.Item>
                <List.Item>Cache efficiency and memory access patterns</List.Item>
              </List>
            </div>

            {/* Part 4 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 4: Broadcasting Rules Implementation</Title>
              <Text className="text-gray-700 mb-4">
                Implement and understand broadcasting mechanics manually:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Write a function to predict broadcasting compatibility</List.Item>
                <List.Item>Implement custom broadcasting for complex operations</List.Item>
                <List.Item>Analyze memory implications of broadcasting</List.Item>
                <List.Item>Create broadcasting-aware custom operations</List.Item>
              </List>
            </div>

            {/* Part 5 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 5: Storage and Stride Analysis</Title>
              <Text className="text-gray-700 mb-4">
                Deep dive into PyTorch's memory model:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Visualize tensor storage and stride patterns</List.Item>
                <List.Item>Create custom tensor layouts using as_strided</List.Item>
                <List.Item>Analyze performance implications of different memory layouts</List.Item>
                <List.Item>Implement memory-efficient sliding window operations</List.Item>
              </List>
            </div>

            {/* Part 6 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 6: Advanced Linear Algebra</Title>
              <Text className="text-gray-700 mb-4">
                Implement and benchmark advanced linear algebra operations:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Custom implementation of matrix decompositions (QR, SVD)</List.Item>
                <List.Item>Eigenvalue computation and spectral analysis</List.Item>
                <List.Item>Numerical stability analysis of different algorithms</List.Item>
                <List.Item>Batched linear algebra operations for efficiency</List.Item>
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

export default Exercise3;