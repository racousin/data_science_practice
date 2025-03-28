import React from 'react';
import { Stack, Text, Title, Paper, Code } from '@mantine/core';

export default function ConvolutionExplanation() {
  // Sample kernel and input matrix for visualization
  const kernel = [
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
  ];
  
  const inputMatrix = [
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
    [130, 140, 150, 160]
  ];

  return (
    <Stack className="w-full max-w-4xl space-y-6 p-6">
      <Title className="text-2xl font-bold">Understanding Convolutions</Title>
      
      <Paper className="p-6 bg-gray-50">
        <Stack className="space-y-4">
          <Text className="font-semibold">Definition:</Text>
          <Text>
            A 2D convolution is a mathematical operation that computes a weighted sum between 
            two matrices: an input matrix (typically an image or feature map) and a kernel 
            (also called a filter). The operation is defined as:
          </Text>
          
          <div className="p-4 bg-white rounded-md">
            <Text className="font-mono text-sm">
              {`G[i,j] = ∑∑ F[i+k,j+l] * K[k,l]`}
            </Text>
            <Text className="text-sm text-gray-600 mt-2">
              where:
            </Text>
            <ul className="list-disc ml-6 text-sm text-gray-600">
              <li>G[i,j] is the output at position (i,j)</li>
              <li>F is the input matrix</li>
              <li>K is the kernel matrix</li>
              <li>k,l iterate over kernel dimensions</li>
            </ul>
          </div>

          <Title className="text-xl mt-4">Step-by-Step Process:</Title>
          
          <Stack className="space-y-2">
            <Text className="font-semibold">1. Kernel Placement:</Text>
            <Text>
              The kernel is positioned over a region of the input matrix, starting from the top-left.
            </Text>

            <Text className="font-semibold">2. Element-wise Multiplication:</Text>
            <Text>
              Each element of the kernel is multiplied with its corresponding input element.
            </Text>

            <Text className="font-semibold">3. Summation:</Text>
            <Text>
              All products are summed to produce a single output value.
            </Text>

            <Text className="font-semibold">4. Sliding:</Text>
            <Text>
              The kernel slides across the input matrix (typically with a stride of 1), repeating steps 2-3.
            </Text>
          </Stack>

          <Paper className="p-4 bg-white mt-4">
            <Title className="text-lg">Example Calculation</Title>
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div>
                <Text className="font-semibold mb-2">Input Matrix:</Text>
                <div className="grid grid-cols-4 gap-1">
                  {inputMatrix.map((row, i) => (
                    row.map((val, j) => (
                      <div key={`input-${i}-${j}`} className="w-8 h-8 bg-blue-50 flex items-center justify-center text-sm border">
                        {val}
                      </div>
                    ))
                  ))}
                </div>
              </div>
              
              <div>
                <Text className="font-semibold mb-2">Kernel:</Text>
                <div className="grid grid-cols-3 gap-1">
                  {kernel.map((row, i) => (
                    row.map((val, j) => (
                      <div key={`kernel-${i}-${j}`} className="w-8 h-8 bg-green-50 flex items-center justify-center text-sm border">
                        {val}
                      </div>
                    ))
                  ))}
                </div>
              </div>
              
              <div>
                <Text className="font-semibold mb-2">Output Region:</Text>
                <div className="grid grid-cols-2 gap-1">
                  {[...Array(4)].map((_, i) => (
                    <div key={`output-${i}`} className="w-8 h-8 bg-gray-100 flex items-center justify-center text-sm border">
                      {i === 0 ? '*' : ''}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Paper>

          <Stack className="space-y-2 mt-4">
            <Text className="font-semibold">Key Properties:</Text>
            <ul className="list-disc ml-6">
              <li>Translation Equivariance: Pattern detection is position-independent</li>
              <li>Parameter Sharing: Same kernel weights applied across the entire input</li>
              <li>Local Connectivity: Each output depends only on nearby inputs</li>
              <li>Dimensionality: Output size = (N - K + 2P)/S + 1, where:
                <ul className="list-disc ml-6 mt-1">
                  <li>N = input size</li>
                  <li>K = kernel size</li>
                  <li>P = padding</li>
                  <li>S = stride</li>
                </ul>
              </li>
            </ul>
          </Stack>
        </Stack>
      </Paper>
    </Stack>
  );
}