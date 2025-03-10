import React from 'react';
import { Text, Stack, Table, Code } from '@mantine/core';
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

      <Text>
        In PyTorch, this operation is implemented efficiently as a simple lookup:
      </Text>
      <BlockMath>embedded\_vector = W[i]</BlockMath>

      <Text>
        During training, the weight matrix <InlineMath>W</InlineMath> is learned through backpropagation, optimizing the parameters to minimize the overall loss function <InlineMath>L</InlineMath>:
      </Text>
      <BlockMath math="\frac{\partial L}{\partial W} = \frac{\partial L}{\partial E(i)} \frac{\partial E(i)}{\partial W}" />

      <Text>
        The gradient update only affects the specific rows of <InlineMath>W</InlineMath> that were accessed during the forward pass:
      </Text>
      <BlockMath math="W_i \leftarrow W_i - \alpha \frac{\partial L}{\partial E(i)}" />
      
      <Text>
        where <InlineMath>\alpha</InlineMath> is the learning rate. This sparse update is computationally efficient and allows the model to learn meaningful representations.
      </Text>

      <Text>
        A key property of learned embeddings is that semantically similar categories tend to have similar vector representations in the embedding space, as measured by cosine similarity:
      </Text>
      <BlockMath math="similarity(E(i), E(j)) = \frac{E(i) \cdot E(j)}{||E(i)|| \cdot ||E(j)||}" />

      <Text weight={500} mt="md">Determining Embedding Dimensions:</Text>
      <Text>
        The embedding dimension <InlineMath>d</InlineMath> is a hyperparameter that balances representational capacity against overfitting. A common rule of thumb is:
      </Text>
      <BlockMath math="d = min(50, \lfloor 1.6 \times n^{0.56} \rfloor)" />
      
      <Table striped highlightOnHover mt="sm">
        <thead>
          <tr>
            <th>Category Cardinality</th>
            <th>Recommended Embedding Dim</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {embeddingSizes.map((size, index) => (
            <tr key={index}>
              <td>{size.cardinality}</td>
              <td>{size.recommendedDim}</td>
              <td>{size.note}</td>
            </tr>
          ))}
        </tbody>
      </Table>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

# Example: Creating an embedding layer for user_id with 1000 unique users
num_users = 1000
embedding_dim = 16

# Create embedding layer
user_embedding = nn.Embedding(
    num_embeddings=num_users,
    embedding_dim=embedding_dim
)

# Convert categorical IDs to embeddings
user_ids = torch.tensor([1, 42, 123])  # Example user IDs
embedded_users = user_embedding(user_ids)
print(f"Shape of embedded users: {embedded_users.shape}")  # torch.Size([3, 16])
`}
      />

      <Text weight={500} mt="md">Practical Example: Multi-Feature Model</Text>
      <CodeBlock
        language="python"
        code={`
class CategoryEmbeddingModel(nn.Module):
    def __init__(self, cat_dims, num_numerical, embedding_dims, hidden_dims):
        super().__init__()
        
        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim) 
            for cat_dim, emb_dim in zip(cat_dims, embedding_dims)
        ])
        
        # Calculate total input size after embeddings
        total_emb_dim = sum(embedding_dims)
        input_dim = total_emb_dim + num_numerical
        
        # Create fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], 1)
        )
    
    def forward(self, cat_x, num_x):
        # Process categorical features through embeddings
        embedded = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        
        # Concatenate with numerical features
        x = torch.cat([embedded, num_x], dim=1)
        
        return self.fc_layers(x)

# Usage example
cat_dims = [1000, 500, 100]  # Cardinality of categorical variables
embedding_dims = [16, 12, 8]  # Corresponding embedding dimensions
num_numerical = 10  # Number of numerical features
hidden_dims = [128, 64]  # Hidden layer dimensions

model = CategoryEmbeddingModel(
    cat_dims=cat_dims,
    num_numerical=num_numerical,
    embedding_dims=embedding_dims,
    hidden_dims=hidden_dims
)
`}
      />
    </Stack>
  );
};

export default CategoricalEmbeddings;