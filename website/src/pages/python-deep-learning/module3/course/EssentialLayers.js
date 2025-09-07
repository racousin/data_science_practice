import React from 'react';
import { Title, Text, Stack, Accordion, Container, List, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

// Import components
import Activation from './EssentialComponents/Activation';
import Dropout from './EssentialComponents/Dropout';
import CategoricalEmbeddings from './EssentialComponents/CategoricalEmbeddings';
import BatchNormalization from './EssentialComponents/BatchNormalization';
import SkipConnections from './EssentialComponents/ResidualConnections';

const EssentialLayers = () => {
  return (
    <Container fluid>
      <Stack spacing="lg">
        <div data-slide>
        <Title order={1}>Essential Layers</Title>
        </div>
        <div data-slide>
          <Title order={2}>Foundation Layers</Title>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Linear/Fully Connected</Title>
          <Text>
            Core transformation layer performing matrix multiplication + bias.
          </Text>
          <Text mt="xs"><strong>Math Formulation:</strong></Text>
          <BlockMath math="y = xW^T + b" />
          <Text>where <InlineMath math="x \in \mathbb{R}^{n \times d_{in}}" />, <InlineMath math="W \in \mathbb{R}^{d_{out} \times d_{in}}" />, <InlineMath math="b \in \mathbb{R}^{d_{out}}" /></Text>
          <Text mt="xs"><strong>Input/Output Shape:</strong></Text>
          <List>
            <List.Item>Input: <InlineMath math="(batch\_size, in\_features)" /></List.Item>
            <List.Item>Output: <InlineMath math="(batch\_size, out\_features)" /></List.Item>
          </List>
          <Text mt="xs"><strong>Parameters:</strong> <InlineMath math="in\_features \times out\_features + out\_features" /></Text>
          <CodeBlock language="python" code={`nn.Linear(in_features=128, out_features=64)
# Performs: output = input @ weight.T + bias`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Convolution</Title>
          <Text>
            Spatial pattern detection in images. Local connectivity with shared weights.
          </Text>
          <Text mt="xs"><strong>Math Formulation (2D Conv):</strong></Text>
          <BlockMath math="y_{i,j,c} = \sum_{m,n,k} x_{i \cdot s + m, j \cdot s + n, k} \cdot W_{m,n,k,c} + b_c" />
          <Text>where <InlineMath math="s" /> is stride, <InlineMath math="W" /> is the kernel weights</Text>
          <Text mt="xs"><strong>Input/Output Shape (Conv2d):</strong></Text>
          <List>
            <List.Item>Input: <InlineMath math="(batch, in\_channels, height, width)" /></List.Item>
            <List.Item>Output: <InlineMath math="(batch, out\_channels, H_{out}, W_{out})" /></List.Item>
          </List>
          <Text>where <InlineMath math="H_{out} = \lfloor \frac{H + 2p - k}{s} \rfloor + 1" />, <InlineMath math="W_{out} = \lfloor \frac{W + 2p - k}{s} \rfloor + 1" /></Text>
          <Text mt="xs"><strong>Parameters:</strong> <InlineMath math="(kernel\_size^2 \times in\_channels + 1) \times out\_channels" /></Text>
          <CodeBlock language="python" code={`nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# Slides kernel across input, computing local feature maps`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Recurrent</Title>
          <Text>
            Process sequential data by maintaining hidden state across time steps.
          </Text>
          <Text mt="xs"><strong>Math Formulation (LSTM):</strong></Text>
          <BlockMath math="f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)" />
          <BlockMath math="i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)" />
          <BlockMath math="o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)" />
          <BlockMath math="c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)" />
          <BlockMath math="h_t = o_t \odot \tanh(c_t)" />
          <Text mt="xs"><strong>Input/Output Shape (LSTM):</strong></Text>
          <List>
            <List.Item>Input: <InlineMath math="(seq\_len, batch, input\_size)" /></List.Item>
            <List.Item>Output: <InlineMath math="(seq\_len, batch, hidden\_size)" /></List.Item>
            <List.Item>Hidden: <InlineMath math="(num\_layers, batch, hidden\_size)" /></List.Item>
          </List>
          <Text mt="xs"><strong>Parameters per layer:</strong> <InlineMath math="4 \times (input\_size \times hidden\_size + hidden\_size^2 + 2 \times hidden\_size)" /></Text>
          <CodeBlock language="python" code= {`nn.LSTM(input_size=128, hidden_size=256, num_layers=2)
# Process sequences with memory of previous inputs`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Attention</Title>
          <Text>
            Dynamic focus mechanism that relates different positions in sequences.
          </Text>
          <Text mt="xs"><strong>Math Formulation (Scaled Dot-Product Attention):</strong></Text>
          <BlockMath math="Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V" />
          <Text>For Multi-Head Attention:</Text>
          <BlockMath math="MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O" />
          <BlockMath math="head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)" />
          <Text mt="xs"><strong>Input/Output Shape:</strong></Text>
          <List>
            <List.Item>Query: <InlineMath math="(seq\_len, batch, embed\_dim)" /></List.Item>
            <List.Item>Key: <InlineMath math="(seq\_len, batch, embed\_dim)" /></List.Item>
            <List.Item>Value: <InlineMath math="(seq\_len, batch, embed\_dim)" /></List.Item>
            <List.Item>Output: <InlineMath math="(seq\_len, batch, embed\_dim)" /></List.Item>
          </List>
          <Text mt="xs"><strong>Parameters:</strong> <InlineMath math="4 \times embed\_dim^2" /> (for Q, K, V, and output projections)</Text>
          <CodeBlock language="python" code={`nn.MultiheadAttention(embed_dim=512, num_heads=8)
# Computes weighted importance between sequence elements`}/>
        </div>

        
          
          
          <Activation/>
        
          <Title order={2}>Regularization Layers</Title>
          
          
          <Dropout/>
          
<BatchNormalization/>
          <Title order={2}>Specialized Components</Title>
          
          
<CategoricalEmbeddings/>
          
          
<SkipConnections/>
        
      </Stack>
    </Container>
  );
};
export default EssentialLayers;