import React from 'react';
import { Title, Text, Stack, List, Alert, Divider, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const BatchNormalization = () => {
  return (
    
    <Stack spacing="md">
      <div data-slide>
      <Title order={3} mt="md">Batch Normalization</Title>
      <Text>
        Batch Normalization (BatchNorm) is a technique introduced by Sergey Ioffe and Christian Szegedy in 2015
        that normalizes the intermediate activations of neural networks, significantly improving training 
        stability and speed. It addresses the internal covariate shift problem by normalizing layer inputs.
      </Text>
      <CodeBlock
        language="python"
        code={`self.bn1 = nn.BatchNorm1d(256)`}
      />
      </div>
      <div data-slide>
      <Text>
        Consider a mini-batch of activations at some layer: <InlineMath>{`\\mathcal{B} = \\{x_1, x_2, ..., x_m\\}`}</InlineMath>
      </Text>

      <Text>
        First, we calculate the mean and variance of the mini-batch:
      </Text>
      <BlockMath>
        {`\\mu_\\mathcal{B} = \\frac{1}{m}\\sum_{i=1}^m x_i`}
      </BlockMath>
      <BlockMath>
        {`\\sigma_\\mathcal{B}^2 = \\frac{1}{m}\\sum_{i=1}^m (x_i - \\mu_\\mathcal{B})^2`}
      </BlockMath>

      <Text>
        Next, we normalize each activation:
      </Text>
      <BlockMath>
        {`\\hat{x}_i = \\frac{x_i - \\mu_\\mathcal{B}}{\\sqrt{\\sigma_\\mathcal{B}^2 + \\epsilon}}`}
      </BlockMath>
</div>
<div data-slide>
      <Title order={3}>Training vs. Inference</Title>
      <Text>
        During training, for each batch:
      </Text>
      <List>
        <List.Item>We calculate and use that batch's mean and variance for normalization</List.Item>
        <List.Item>We update running estimates of the global statistics:</List.Item>
      </List>
      <CodeBlock
        language="python"
        code={`# Training mode: use mini-batch statistics
model.train()
train_predictions = model(train_data)`}
      />

      <Text>
        During inference, we apply the global statistics. 
        Here we use the stored running statistics:
      </Text>
      <CodeBlock
        language="python"
        code={`# Inference mode: use running statistics
model.eval()
test_predictions = model(test_data)`}
      />
</div>
    </Stack>
  );
};

export default BatchNormalization;