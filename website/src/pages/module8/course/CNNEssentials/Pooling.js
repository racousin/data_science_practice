import React from 'react';
import { Text, Stack, Code, Image, Table, Grid, List, Title, Accordion } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';


const PoolingExample = ({ title, image, params, code }) => (
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
        <Code block className="mb-2">{code}</Code>
      </Stack>
    </Accordion.Panel>
  </Accordion.Item>
);

const PoolingExamples = () => {
  const examples = [
    {
      title: "Basic Max Pooling",
      image: { src: "/assets/module8/maxpool.png", alt: "maxpool" },
      params: {
        "Pool Size": "2 × 2",
        "Input Size": "4 × 4",
        "Stride": "2",
        "Output Size": "2 × 2 (⌊(4 - 2)/2⌋ + 1 = 2)",
        "Operation": "Takes maximum value in each 2×2 region"
      },
      code: "nn.MaxPool2d(kernel_size=2, stride=2)"
    },
    {
      title: "Global Average Pooling",
      image: { src: "/assets/module8/globalpool.png", alt: "globalpool" },
      params: {
        "Pool Size": "Input size",
        "Input Size": "4 × 4",
        "Output Size": "1 × 1",
        "Operation": "Averages entire feature map into single value"
      },
      code: "nn.AdaptiveAvgPool2d(output_size=1)"
    },
    {
      title: "Overlapping Max Pooling",
      image: { src: "/assets/module8/overlappool.png", alt: "overlappool" },
      params: {
        "Pool Size": "3 × 3",
        "Input Size": "5 × 5",
        "Stride": "2",
        "Output Size": "2 × 2 (⌊(5 - 3)/2⌋ + 1 = 2)",
        "Operation": "Max pooling with overlapping windows"
      },
      code: "nn.MaxPool2d(kernel_size=3, stride=2)"
    }
  ];

  return (
    <Stack spacing="xl">
      <Accordion variant="contained">
        {examples.map((example) => (
          <PoolingExample key={example.title} {...example} />
        ))}
      </Accordion>
    </Stack>
  );
};

const Pooling = () => {

  return (
    <Stack spacing="md">
      <Text>
        Pooling layers are essential components in CNNs that reduce spatial dimensions
        while retaining important features. They help achieve:
      </Text>
      
      <List>
        <List.Item>Translation invariance</List.Item>
        <List.Item>Spatial hierarchy of features</List.Item>
        <List.Item>Computational efficiency</List.Item>
        <List.Item>Control of overfitting</List.Item>
      </List>

      <Text weight={700}>1. Common Pooling Operations</Text>

      <Grid>
        <Grid.Col span={12} md={6}>
          <Text>Max Pooling:</Text>
          <BlockMath>
            {`MaxPool(X)_{i,j} = \\max_{(m,n) \\in R_{i,j}} x_{m,n}`}
          </BlockMath>
        </Grid.Col>
        
        <Grid.Col span={12} md={6}>
          <Text>Average Pooling:</Text>
          <BlockMath>
            {`AvgPool(X)_{i,j} = \\frac{1}{|R_{i,j}|} \\sum_{(m,n) \\in R_{i,j}} x_{m,n}`}
          </BlockMath>
        </Grid.Col>
      </Grid>

      <Text>
        Where <InlineMath>{`R_{i,j}`}</InlineMath> represents the pooling region centered
        at position <InlineMath>(i,j)</InlineMath>.
      </Text>
      <Image src="/assets/module8/pooling.png" alt="pooling" />
      <Text weight={700}>2. Implementation Examples</Text>
      

      <Text>
        The output size after pooling can be calculated using:
      </Text>
      
      <BlockMath>
        {`O = \\left\\lfloor\\frac{N - K}{S}\\right\\rfloor + 1`}
      </BlockMath>
      
      <Text>
        Where <InlineMath>N</InlineMath> is input size, <InlineMath>K</InlineMath> is kernel size,
        and <InlineMath>S</InlineMath> is stride. Unlike convolution, pooling typically doesn't
        use padding as we want to reduce spatial dimensions.
      </Text>
      <Title order={5} className="mb-2">Note about Adaptive Pooling:</Title>
      <Text>
  In PyTorch's AdaptiveAvgPool2d/AdaptiveMaxPool2d:
</Text>
<List>
  <List.Item>
    Unlike traditional pooling, you specify the desired output size directly instead of kernel size and stride
  </List.Item>
  <List.Item>
    The layer automatically calculates the necessary parameters to achieve the target output dimensions
  </List.Item>
  <List.Item>
    Particularly useful when dealing with varying input sizes or when you need a specific output dimension for downstream tasks
  </List.Item>
  <List.Item>
    Common use case: adapting feature maps to a fixed size before feeding them into fully connected layers
  </List.Item>
</List>

<Text mt="md">
  For example, using <Code>nn.AdaptiveAvgPool2d((1, 1))</Code> will reduce any input feature map to a single value per channel, regardless of the input dimensions.
</Text>

    </Stack>
  );
};

export default Pooling;