import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';

const Transformers = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Transformer Architecture
          </Title>
          <Text size="xl" className="text-center mb-6">
            Attention Is All You Need
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                Transformers revolutionized deep learning by replacing recurrent connections with 
                self-attention mechanisms. They enable parallel processing of sequences and have 
                become the foundation for modern language models and vision architectures.
              </Text>
              <List>
                <List.Item>Multi-head self-attention mechanism</List.Item>
                <List.Item>Positional encoding and embeddings</List.Item>
                <List.Item>Encoder-decoder architecture</List.Item>
                <List.Item>Modern variants and applications</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Attention Mechanism */}
        <div data-slide className="min-h-[500px]" id="attention-mechanism">
          <Title order={2} mb="xl">Self-Attention Mechanism</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Text size="lg">
              Self-attention allows each position in a sequence to attend to all other positions,
              capturing long-range dependencies without the sequential processing constraints of RNNs.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Scaled Dot-Product Attention</Title>
                <Code block language="python">{`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, n_heads, seq_len, d_k)
        
        # Compute attention scores
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, v)
        
        return output, attn

def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Multi-Head Attention */}
        <div data-slide className="min-h-[500px]" id="multi-head-attention">
          <Title order={2} mb="xl">Multi-Head Attention</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-green-50 mb-4">
                <Title order={4} mb="sm">Multi-Head Attention Implementation</Title>
                <Code block language="python">{`class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_heads = self.d_k, self.d_k, self.n_heads
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_heads, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_heads, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_heads, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Simplified Multi-Head Attention</Title>
                <Code block language="python">{`class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} mb="sm">Cross-Attention</Title>
                <Code block language="python">{`class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key_value):
        # query: decoder input
        # key_value: encoder output
        B, N_q, C = query.shape
        _, N_kv, _ = key_value.shape
        
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        return self.out_proj(out)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: Positional Encoding */}
        <div data-slide className="min-h-[500px]" id="positional-encoding">
          <Title order={2} mb="xl">Positional Encoding</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} mb="sm">Sinusoidal Positional Encoding</Title>
                <Code block language="python">{`class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Alternative implementation
def get_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(math.log(10000.0) / d_model))
    
    pos_encoding = torch.zeros(seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">Learnable Positional Embedding</Title>
                <Code block language="python">{`class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embedding(pos)

# Relative positional encoding
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_relative_position):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.randn(max_relative_position * 2 + 1, d_model)
        )
        
    def forward(self, length):
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - range_mat.T
        
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position,
            self.max_relative_position
        )
        
        final_mat = distance_mat_clipped + self.max_relative_position
        embeddings = self.embeddings_table[final_mat]
        
        return embeddings`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 5: Transformer Block */}
        <div data-slide className="min-h-[500px]" id="transformer-block">
          <Title order={2} mb="xl">Transformer Block Architecture</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-indigo-50 mb-4">
                <Title order={4} mb="sm">Encoder Block</Title>
                <Code block language="python">{`class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_len, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Embedding + positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Decoder Block</Title>
                <Code block language="python">{`class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Attention Masks</Title>
                <Code block language="python">{`def create_padding_mask(seq, pad_idx=0):
    """Create mask to ignore padding tokens"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(1)

def create_look_ahead_mask(size):
    """Create mask to prevent attention to future positions"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

def create_masks(src, tgt, pad_idx=0):
    """Create all necessary masks for transformer"""
    src_mask = create_padding_mask(src, pad_idx)
    
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1))
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask
    
    return src_mask, tgt_mask

# Example usage
src = torch.tensor([[1, 2, 3, 4, 0, 0], [1, 2, 0, 0, 0, 0]])  # batch=2, seq=6
tgt = torch.tensor([[1, 2, 3], [1, 2, 3]])  # batch=2, seq=3

src_mask, tgt_mask = create_masks(src, tgt, pad_idx=0)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 6: Complete Transformer */}
        <div data-slide className="min-h-[500px]" id="complete-transformer">
          <Title order={2} mb="xl">Complete Transformer Model</Title>
          
          <Paper className="p-4 bg-gray-50">
            <Code block language="python">{`class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_len=5000, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Encoder
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, num_encoder_layers, d_ff, max_len, dropout
        )
        
        # Decoder
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)
        
        # Decode target sequence
        tgt_embedded = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.decoder_pos_encoding(tgt_embedded)
        tgt_embedded = self.dropout(tgt_embedded)
        
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
        
        # Final output projection
        output = self.output_projection(decoder_output)
        return output

# Model instantiation and usage
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# Example forward pass
src = torch.randint(1, 10000, (32, 100))  # batch_size=32, seq_len=100
tgt = torch.randint(1, 10000, (32, 80))   # batch_size=32, seq_len=80

src_mask, tgt_mask = create_masks(src, tgt)
output = model(src, tgt, src_mask, tgt_mask)
print(f"Output shape: {output.shape}")  # (32, 80, 10000)`}</Code>
          </Paper>
        </div>

        {/* Slide 7: Modern Transformer Variants */}
        <div data-slide className="min-h-[500px]" id="modern-variants">
          <Title order={2} mb="xl">Modern Transformer Variants</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} mb="sm">BERT-style Encoder</Title>
                <Code block language="python">{`class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_len):
        super().__init__()
        
        self.embeddings = nn.ModuleDict({
            'token': nn.Embedding(vocab_size, d_model),
            'position': nn.Embedding(max_len, d_model),
            'token_type': nn.Embedding(2, d_model)  # For NSP task
        })
        
        self.encoder = TransformerEncoder(
            vocab_size, d_model, n_heads, num_layers, d_ff, max_len
        )
        
        # Task heads
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.nsp_head = nn.Linear(d_model, 2)
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Combine embeddings
        embeddings = (self.embeddings['token'](input_ids) + 
                     self.embeddings['position'](position_ids) + 
                     self.embeddings['token_type'](token_type_ids))
        
        # Encode
        encoded = self.encoder(embeddings)
        
        # Task-specific outputs
        mlm_logits = self.mlm_head(encoded)
        nsp_logits = self.nsp_head(encoded[:, 0, :])  # [CLS] token
        
        return mlm_logits, nsp_logits`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">GPT-style Decoder</Title>
                <Code block language="python">{`class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_len):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(pos_ids)
        x = token_embeds + pos_embeds
        
        # Apply blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-4 bg-red-50 mt-4">
            <Title order={4} mb="sm">Vision Transformer (ViT)</Title>
            <Code block language="python">{`class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, d_model=768, 
                 num_layers=12, n_heads=12, d_ff=3072):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Create patches and flatten
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])  # Use class token for classification`}</Code>
          </Paper>
        </div>

        {/* Slide 8: Training and Optimization */}
        <div data-slide className="min-h-[500px]" id="training-optimization">
          <Title order={2} mb="xl">Training Transformers: Best Practices</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Learning Rate Scheduling</Title>
                <Code block language="python">{`class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Label smoothing for training stability
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, pred, target):
        # Create smoothed target distribution
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (self.vocab_size - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return self.criterion(F.log_softmax(pred, dim=1), smooth_target)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Memory Optimization</Title>
                <Code block language="python">{`# Gradient checkpointing for memory efficiency
import torch.utils.checkpoint as checkpoint

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.block = TransformerEncoderBlock(d_model, n_heads, d_ff)
        
    def forward(self, x, mask=None):
        return checkpoint.checkpoint(self.block, x, mask)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model = Transformer(...).cuda()
optimizer = torch.optim.AdamW(model.parameters())

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(batch['src'], batch['tgt'])
        loss = criterion(output, batch['labels'])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Flash Attention for efficiency (conceptual)
try:
    from flash_attn import flash_attn_func
    
    def efficient_attention(q, k, v, causal=False):
        return flash_attn_func(q, k, v, causal=causal)
except ImportError:
    def efficient_attention(q, k, v, causal=False):
        # Fallback to standard attention
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default Transformers;