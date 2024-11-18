import React from 'react';
import { Text, Stack, Table, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';

const CategoricalEmbeddings = () => {
  // Sample data for embedding size table
  const embeddingSizes = [
    { cardinality: '2-4', size: '2' },
    { cardinality: '5-9', size: '3' },
    { cardinality: '10-24', size: '4' },
    { cardinality: '25-49', size: '5' },
    { cardinality: '50+', size: 'min(8, round(1.6 * log(n)))' }
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