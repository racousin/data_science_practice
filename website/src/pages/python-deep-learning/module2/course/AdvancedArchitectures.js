import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';

const AdvancedArchitectures = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Advanced Neural Network Architectures
          </Title>
          <Text size="xl" className="text-center mb-6">
            Beyond Basic MLPs: Modern Network Designs
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                Modern deep learning employs sophisticated architectures that go beyond simple feedforward networks.
                These architectures incorporate specialized components and design patterns for improved performance.
              </Text>
              <List>
                <List.Item>Residual connections and skip connections</List.Item>
                <List.Item>Attention mechanisms and self-attention</List.Item>
                <List.Item>Batch normalization and layer normalization</List.Item>
                <List.Item>Advanced activation functions and gating</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Residual Networks */}
        <div data-slide className="min-h-[500px]" id="residual-networks">
          <Title order={2} mb="xl">Residual Networks (ResNets)</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Text size="lg">
              Residual connections allow gradients to flow directly through skip connections,
              enabling training of very deep networks and mitigating the vanishing gradient problem.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Basic Residual Block</Title>
                <Code block language="python">{`import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # Skip connection projection if needed
        self.skip_connection = None
        if in_features != out_features:
            self.skip_connection = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        identity = x
        
        if self.skip_connection is not None:
            identity = self.skip_connection(x)
        
        out = self.layers(x)
        out += identity  # Residual connection
        return torch.relu(out)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">ResNet Architecture</Title>
                <Code block language="python">{`class ResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512),
        )
        
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        return self.classifier(x)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Attention Mechanisms */}
        <div data-slide className="min-h-[500px]" id="attention-mechanisms">
          <Title order={2} mb="xl">Attention Mechanisms</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-purple-50 mb-4">
                <Title order={4} mb="sm">Self-Attention Implementation</Title>
                <Code block language="python">{`class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(context)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} mb="sm">Multi-Head Attention Block</Title>
                <Code block language="python">{`class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        attended = self.attention(x)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} mb="sm">Transformer Architecture</Title>
                <Code block language="python">{`class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, 
                 num_layers, ff_dim, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: Normalization Techniques */}
        <div data-slide className="min-h-[500px]" id="normalization">
          <Title order={2} mb="xl">Normalization Techniques</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Batch Normalization</Title>
                <Code block language="python">{`# Batch Normalization
class BatchNormMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Layer Normalization</Title>
                <Code block language="python">{`# Layer Normalization
class LayerNormMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-4 bg-purple-50 mt-4">
            <Title order={4} mb="sm">Group Normalization</Title>
            <Code block language="python">{`# Group Normalization (typically used in CNNs, but can be adapted)
class GroupNorm1d(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        # x: (N, C) -> (N, num_groups, C // num_groups)
        N, C = x.shape
        x = x.view(N, self.num_groups, C // self.num_groups)
        
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(N, C)
        
        return x * self.weight + self.bias`}</Code>
          </Paper>
        </div>

        {/* Slide 5: Advanced Activation Functions */}
        <div data-slide className="min-h-[500px]" id="activations">
          <Title order={2} mb="xl">Advanced Activation Functions</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} mb="sm">Modern Activations</Title>
                <Code block language="python">{`# GELU (Gaussian Error Linear Unit)
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / 3.14159)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

# Swish/SiLU
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Mish
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">Gated Linear Units</Title>
                <Code block language="python">{`class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2 * output_dim)
    
    def forward(self, x):
        # Split into two halves
        x1, x2 = self.linear(x).chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)

class GeGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2 * output_dim)
    
    def forward(self, x):
        x1, x2 = self.linear(x).chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 6: Architecture Design Principles */}
        <div data-slide className="min-h-[500px]" id="design-principles">
          <Title order={2} mb="xl">Architecture Design Principles</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Key Design Principles</Title>
                <List>
                  <List.Item><strong>Skip Connections:</strong> Enable gradient flow in deep networks</List.Item>
                  <List.Item><strong>Normalization:</strong> Stabilize training and reduce internal covariate shift</List.Item>
                  <List.Item><strong>Attention:</strong> Allow selective focus on relevant information</List.Item>
                  <List.Item><strong>Bottlenecks:</strong> Reduce computational complexity while preserving information</List.Item>
                  <List.Item><strong>Regularization:</strong> Prevent overfitting through architectural constraints</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Implementation Tips</Title>
                <List>
                  <List.Item>Start simple and add complexity gradually</List.Item>
                  <List.Item>Use proven architectural patterns as building blocks</List.Item>
                  <List.Item>Consider computational efficiency and memory usage</List.Item>
                  <List.Item>Experiment with different normalization placements</List.Item>
                  <List.Item>Balance model capacity with regularization</List.Item>
                  <List.Item>Profile your architecture for bottlenecks</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 7: Complete Example */}
        <div data-slide className="min-h-[500px]" id="complete-example">
          <Title order={2} mb="xl">Complete Modern Architecture Example</Title>
          
          <Paper className="p-4 bg-gray-50">
            <Code block language="python">{`class ModernMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, 
                 num_heads=8, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Residual blocks with attention
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.blocks.append(
                nn.ModuleDict({
                    'attention': SelfAttention(hidden_sizes[i], num_heads),
                    'norm1': nn.LayerNorm(hidden_sizes[i]),
                    'ffn': nn.Sequential(
                        nn.Linear(hidden_sizes[i], hidden_sizes[i] * 4),
                        GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_sizes[i] * 4, hidden_sizes[i+1])
                    ),
                    'norm2': nn.LayerNorm(hidden_sizes[i]),
                    'skip_proj': nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) 
                                if hidden_sizes[i] != hidden_sizes[i+1] else None
                })
            )
        
        # Output head
        self.output_norm = nn.LayerNorm(hidden_sizes[-1])
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Input processing
        x = self.input_norm(self.input_proj(x))
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        
        # Process through residual blocks
        for block in self.blocks:
            residual = x
            
            # Self-attention
            attended = block['attention'](x)
            x = block['norm1'](x + attended)
            
            # Feed-forward network
            ff_out = block['ffn'](x)
            
            # Skip connection with projection if needed
            if block['skip_proj'] is not None:
                residual = block['skip_proj'](residual)
            
            x = block['norm2'](residual + ff_out)
        
        # Output
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.output_norm(x)
        x = self.dropout(x)
        return self.classifier(x)`}</Code>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default AdvancedArchitectures;