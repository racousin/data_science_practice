import React from 'react';
import { Text, Stack, Table, Code, Title} from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';

const CategoricalEmbeddings = () => {
  // Sample data for embedding size table
  const embeddingSizes = [
    { cardinality: '10', recommendedDim: '5-10', note: 'Small categorical space' },
    { cardinality: '100', recommendedDim: '8-16', note: 'Medium categorical space' },
    { cardinality: '1,000', recommendedDim: '16-32', note: 'Large categorical space' },
    { cardinality: '10,000+', recommendedDim: '32-64', note: 'Very large categorical space' },
  ];

  return (
    <Stack spacing="md">
      <div data-slide>
      <Title order={3} mt="md">Embeddings</Title>
      <Text>
        Handling categorical variables effectively is crucial in deep learning. While traditional methods like one-hot encoding 
        work for small categorical spaces, they become inefficient with high cardinality. Neural network embeddings offer a 
        powerful alternative, learning dense vector representations of categorical data.
      </Text>

      <Stack spacing="xs">
        <Text weight={500}>Key Concepts:</Text>
        <Text>
          1. An embedding layer transforms categorical indices into dense vectors of fixed size
        </Text>
        <Text>
          2. The embedding dimension <InlineMath>d</InlineMath> is a hyperparameter, typically smaller than the number of categories
        </Text>
        <Text>
          3. Embeddings are learned during training, capturing semantic relationships between categories
        </Text>
      </Stack>
</div>
<div data-slide>
      <Text>
        Let <InlineMath>V</InlineMath> be a vocabulary of categorical values with cardinality <InlineMath>|V| = n</InlineMath>. 
        An embedding layer can be formalized as a mapping function:
      </Text>
      <BlockMath math="E: \{1,2,...,n\} \rightarrow \mathbb{R}^d" />

      <Text>
        This mapping is implemented as a lookup in an embedding matrix <InlineMath math="W \in \mathbb{R}^{n \times d}" />, where:
      </Text>
      <BlockMath math="W = \begin{bmatrix} 
        w_{1,1} & w_{1,2} & \cdots & w_{1,d} \\
        w_{2,1} & w_{2,2} & \cdots & w_{2,d} \\
        \vdots & \vdots & \ddots & \vdots \\
        w_{n,1} & w_{n,2} & \cdots & w_{n,d}
        \end{bmatrix}" />

      <Text>
        For a categorical item with index <InlineMath>i</InlineMath>, the embedding operation is:
      </Text>
      <BlockMath math="E(i) = W_i = [w_{i,1}, w_{i,2}, ..., w_{i,d}]" />

      <CodeBlock
        language="python"
        code={`nn.Embedding(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim)`}
      /></div>

    </Stack>
  );
};

export default CategoricalEmbeddings;