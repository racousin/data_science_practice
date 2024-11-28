import React from 'react';
import { Text, Stack, Code, Image, Table, Title, Accordion } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';


const ConvolutionExample = ({ title, image, params, code }) => (
  <Accordion.Item value={title}>
    <Accordion.Control>
      <Title order={5}>{title}</Title>
    </Accordion.Control>
    <Accordion.Panel>
      <Stack spacing="md">
        <Image src={image.src} alt={image.alt} />
        <Table className="mb-4">
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(params).map(([key, value]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>{value}</td>
              </tr>
            ))}
          </tbody>
        </Table>
        <Code block className="mb-2">
          {code}
        </Code>
      </Stack>
    </Accordion.Panel>
  </Accordion.Item>
);

const ConvolutionExamples = () => {
  const examples = [
    {
      title: "Basic 2x2 Convolution",
      image: { src: "/assets/module8/conv_kern1.png", alt: "conv_kern1" },
      params: {
        "Kernel Size": "2 × 2",
        "Input Size": "6 × 6",
        "Channels (in/out)": "1/1",
        "Stride": "1",
        "Padding": "0",
        "Output Size": "5 × 5 (⌊(6 - 2 + 0)/1⌋ + 1 = 5)",
        "Parameters": "4 weights + 1 bias = 5 parameters"
      },
      code: "nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)"
    },
    {
      title: "Strided Convolution",
      image: { src: "/assets/module8/conv_kern2.png", alt: "conv_kern2" },
      params: {
        "Kernel Size": "2 × 2",
        "Input Size": "6 × 6",
        "Channels (in/out)": "1/1",
        "Stride": "2",
        "Padding": "0",
        "Output Size": "3 × 3 (⌊(6 - 2 + 0)/2⌋ + 1 = 3)",
        "Parameters": "4 weights + 1 bias = 5 parameters"
      },
      code: "nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)"
    },
    {
      title: "Padded Strided Convolution",
      image: { src: "/assets/module8/conv_kern3.png", alt: "conv_kern3" },
      params: {
        "Kernel Size": "2 × 2",
        "Input Size": "6 × 6",
        "Channels (in/out)": "1/1",
        "Stride": "2",
        "Padding": "1",
        "Output Size": "4 × 4 (⌊(6 - 2 + 2)/2⌋ + 1 = 4)",
        "Parameters": "4 weights + 1 bias = 5 parameters"
      },
      code: "nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=1)"
    },
    {
      title: "Multi-channel Output Convolution",
      image: { src: "/assets/module8/conv_c1.gif", alt: "conv_c1" },
      params: {
        "Kernel Size": "3 × 3",
        "Input Size": "7 × 7",
        "Channels (in/out)": "1/4",
        "Stride": "1",
        "Padding": "0",
        "Output Size": "5 × 5 (⌊(7 - 3 + 0)/1⌋ + 1 = 5)",
        "Parameters": "(9 weights × 4 filters) + 4 biases = 40 parameters"
      },
      code: "nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0)"
    },
    {
      title: "Multi-channel Input/Output Convolution",
      image: { src: "/assets/module8/conv_c2.gif", alt: "conv_c2" },
      params: {
        "Kernel Size": "3 × 3",
        "Input Size": "7 × 7",
        "Channels (in/out)": "3/4",
        "Stride": "1",
        "Padding": "0",
        "Output Size": "5 × 5 (⌊(7 - 3 + 0)/1⌋ + 1 = 5)",
        "Parameters": "(9 weights × 3 channels × 4 filters) + 4 biases = 112 parameters"
      },
      code: "nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)"
    }
  ];
  return (
    <Stack spacing="xl">
      <Accordion variant="contained">
        {examples.map((example) => (
          <ConvolutionExample key={example.title} {...example} />
        ))}
      </Accordion>
    </Stack>
  );
};

const ConvolutionBasics = () => {

  return (
    <Stack spacing="md">
      <Text>
        The convolution operation is fundamental to CNNs. It performs a sliding window
        operation across the input, computing element-wise multiplications and sums.
      </Text>

      <BlockMath>
        {`(f * g)[n] = \\sum_{m=-\\infty}^{\\infty} f[m]g[n-m]`}
      </BlockMath>

      <Text>
        Key components of a convolution layer:
      </Text>
      
      <ul>
        <li>
          <strong>Kernel Size:</strong> Defines the spatial extent of the convolution
          (e.g., 3×3, 5×5)
        </li>
        <li>
          <strong>Stride:</strong> Controls how the kernel moves across the input
          (<InlineMath>{"s \\in \\mathbb{N}"}</InlineMath>)
        </li>
        <li>
          <strong>Padding:</strong> Added border around input to control output size
        </li>
        <li>
          <strong>Channels:</strong> Number of input/output feature maps
        </li>
      </ul>

      <Text>
        The output size of a convolution layer can be calculated using:
      </Text>

      <BlockMath>
        {`O = \\left\\lfloor\\frac{N - K + 2P}{S}\\right\\rfloor + 1`}
      </BlockMath>

      <Text>
        where:
      </Text>
      <ul>
        <li><InlineMath>O</InlineMath>: Output size</li>
        <li><InlineMath>N</InlineMath>: Input size</li>
        <li><InlineMath>K</InlineMath>: Kernel size</li>
        <li><InlineMath>P</InlineMath>: Padding</li>
        <li><InlineMath>S</InlineMath>: Stride</li>
      </ul>
      <Text>
      The operation is defined as:
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

<ConvolutionExamples/>
<Title order={5} className="mb-2">Note about Bias:</Title>
          <Text>
            In PyTorch's Conv2d layers, each output channel has one learnable bias term that's added after the convolution operation. The bias is added to every spatial position of the output feature map for that channel. This means:
          </Text>
          <ul className="list-disc ml-6 mt-2">
            <li>The number of bias terms equals the number of output channels</li>
            <li>Bias is applied uniformly across spatial dimensions</li>
            <li>Bias can be disabled using bias=False in the Conv2d constructor if needed</li>
          </ul>
    </Stack>
  );
};

export default ConvolutionBasics;