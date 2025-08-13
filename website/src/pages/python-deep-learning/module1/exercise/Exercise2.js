import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Calculator } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise2 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module1/exercises/exercise2.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module1/exercises/exercise2.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/python-deep-learning/module1/exercises/exercise2.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Calculator size={32} className="text-blue-600" />
            <Title order={1} className="text-2xl font-bold">Exercise 2: PyTorch Operations</Title>
          </div>

          <Stack spacing="lg">
            {/* Part 1 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 1: Element-wise Operations</Title>
              <Text className="text-gray-700 mb-4">
                Perform basic mathematical operations on tensors:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Addition, subtraction, multiplication, and division</List.Item>
                <List.Item>Power operations and square roots</List.Item>
                <List.Item>Trigonometric functions</List.Item>
                <List.Item>Logarithmic and exponential functions</List.Item>
              </List>
            </div>

            {/* Part 2 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 2: Reduction Operations</Title>
              <Text className="text-gray-700 mb-4">
                Apply operations that reduce tensor dimensions:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Sum, mean, and product reductions</List.Item>
                <List.Item>Finding minimum and maximum values</List.Item>
                <List.Item>Computing standard deviation and variance</List.Item>
                <List.Item>Argmax and argmin operations</List.Item>
              </List>
            </div>

            {/* Part 3 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 3: Matrix Operations</Title>
              <Text className="text-gray-700 mb-4">
                Perform linear algebra operations:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Matrix multiplication (matmul, mm, bmm)</List.Item>
                <List.Item>Matrix transpose and permutation</List.Item>
                <List.Item>Matrix inverse and determinant</List.Item>
                <List.Item>Eigenvalues and eigenvectors</List.Item>
              </List>
            </div>

            {/* Part 4 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 4: Broadcasting</Title>
              <Text className="text-gray-700 mb-4">
                Understand and apply broadcasting rules:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Broadcasting fundamentals and rules</List.Item>
                <List.Item>Scalar broadcasting to tensors</List.Item>
                <List.Item>Broadcasting across different dimensions</List.Item>
                <List.Item>Common broadcasting patterns and pitfalls</List.Item>
              </List>
            </div>

            {/* Part 5 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 5: Advanced Indexing</Title>
              <Text className="text-gray-700 mb-4">
                Use advanced indexing techniques:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Boolean masking and conditional selection</List.Item>
                <List.Item>Gather and scatter operations</List.Item>
                <List.Item>Index select and masked operations</List.Item>
                <List.Item>Advanced slicing with multiple indices</List.Item>
              </List>
            </div>

            {/* Part 6 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 6: Concatenation and Stacking</Title>
              <Text className="text-gray-700 mb-4">
                Combine multiple tensors:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Concatenating tensors along existing dimensions</List.Item>
                <List.Item>Stacking tensors along new dimensions</List.Item>
                <List.Item>Splitting and chunking tensors</List.Item>
                <List.Item>Unbinding tensors into lists</List.Item>
              </List>
            </div>

            {/* Part 7 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 7: In-place Operations</Title>
              <Text className="text-gray-700 mb-4">
                Understand memory-efficient operations:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>In-place operations with underscore notation</List.Item>
                <List.Item>Memory implications of in-place vs regular operations</List.Item>
                <List.Item>When to use in-place operations</List.Item>
                <List.Item>Gradient computation considerations</List.Item>
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

export default Exercise2;