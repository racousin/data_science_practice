import React from 'react';
import { Text, Title, List, Table } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

export default function TrainingTransformers() {
  return (
    <div>
      <div data-slide>
        <Title order={1}>Training Transformers</Title>
        <Text mt="md">
          Training transformer models requires careful handling of tokenization, model architecture
          definition, and data preparation for both encoder and decoder configurations.
        </Text>
        <Text mt="md">
          This section covers practical aspects of preparing data and training transformer models
          using modern libraries, building on your custom tokenizer knowledge.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Language Modeling Objectives</Title>
        <Text mt="md">
          Transformer models are trained using different language modeling objectives depending
          on their architecture. The two primary training objectives are Causal Language Modeling
          and Masked Language Modeling.
        </Text>
        <Text mt="md">
          These objectives define how the model learns to predict tokens and determine the
          attention patterns used during training and inference.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Causal Language Modeling (CLM)</Title>
        <Text mt="md">
          Causal Language Modeling predicts the next token given only previous tokens.
          This objective is used by decoder-only models like GPT.
        </Text>

        <Text mt="md">The CLM loss function:</Text>
        <BlockMath>{'L_{\\text{CLM}} = -\\sum_{t=1}^{T} \\log P(w_t \\mid w_1, \\ldots, w_{t-1}; \\theta)'}</BlockMath>

        <Text mt="md">Key characteristics:</Text>
        <List spacing="sm" mt="sm">
          <List.Item>Uses causal (unidirectional) attention mask</List.Item>
          <List.Item>Each position can only attend to previous positions</List.Item>
          <List.Item>Enables autoregressive text generation</List.Item>
          <List.Item>Natural for text completion and generation tasks</List.Item>
        </List>

        <Text mt="md">
          The model learns to predict each token based solely on its preceding context,
          making it suitable for sequential generation tasks.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Masked Language Modeling (MLM)</Title>
        <Text mt="md">
          Masked Language Modeling predicts masked tokens given surrounding bidirectional context.
          This objective is used by encoder-only models like BERT.
        </Text>

        <Text mt="md">The MLM loss function:</Text>
        <BlockMath>{'L_{\\text{MLM}} = -\\sum_{i \\in M} \\log P(w_i \\mid w_{\\setminus M}; \\theta)'}</BlockMath>

        <Text mt="md">
          Where <InlineMath>{'M'}</InlineMath> represents the set of masked positions and{' '}
          <InlineMath>{'w_{\\setminus M}'}</InlineMath> represents all tokens except the masked ones.
        </Text>

        <Text mt="md">Key characteristics:</Text>
        <List spacing="sm" mt="sm">
          <List.Item>Uses bidirectional attention</List.Item>
          <List.Item>Randomly masks approximately 15% of tokens during training</List.Item>
          <List.Item>Model predicts masked tokens using full context</List.Item>
          <List.Item>Better for understanding and representation tasks</List.Item>
        </List>
      </div>


      <div data-slide>
        <Title order={2}>Tokenizer Libraries</Title>
        <Text mt="md">
          Modern transformer libraries provide pre-built tokenizers that handle vocabulary
          management, special tokens, and efficient encoding.
        </Text>

        <Title order={3} mt="lg">Hugging Face Tokenizers</Title>
        <Text mt="md">Primary library for transformer tokenization:</Text>

        <CodeBlock
          language="python"
          code={`from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')`}
        />

        <Text mt="md">Key tokenizer methods:</Text>

        <CodeBlock
          language="python"
          code={`# Encode text to token IDs
ids = tokenizer.encode("Hello world")
print(ids)
# [101, 7592, 2088, 102]
# [CLS] hello world [SEP]`}
        />

        <CodeBlock
          language="python"
          code={`# Decode IDs back to text
text = tokenizer.decode(ids)
print(text)
# "[CLS] hello world [SEP]"`}
        />

        <CodeBlock
          language="python"
          code={`# Full encoding with attention masks and token types
encoded = tokenizer(
    "Hello world",
    padding='max_length',
    max_length=8,
    return_tensors='pt'
)`}
        />

        <Text mt="md">Output structure:</Text>
        <CodeBlock
          language="python"
          code={`print(encoded.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

print(encoded['input_ids'])
# tensor([[  101,  7592,  2088,   102,     0,     0,     0,     0]])

print(encoded['attention_mask'])
# tensor([[    1,     1,     1,     1,     0,     0,     0,     0]])`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Using Your Custom Tokenizer</Title>
        <Text mt="md">
          After building a BPE tokenizer, you have a vocabulary mapping tokens to IDs.
          Integrate it with Hugging Face's framework for transformer compatibility.
        </Text>

        <Title order={3} mt="lg">Example Custom Vocabulary</Title>
        <Text mt="md">Your trained tokenizer produces a vocabulary dictionary:</Text>
        <CodeBlock
          language="python"
          code={`# Example vocabulary from your BPE training
vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "hello": 5,
    "world": 6,
    "the": 7
}`}
        />

        <CodeBlock
          language="python"
          code={`    "##ing": 8,    # Subword token
    "transform": 9,
    "##er": 10,
    # ... thousands more tokens
}
# Total vocab_size: 30000`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Integrating Custom Vocabulary</Title>
        <Text mt="md">
          Save your vocabulary and merge files, then load with PreTrainedTokenizerFast.
        </Text>

        <CodeBlock
          language="python"
          code={`import json

# Save vocabulary to file
with open("vocab.json", "w") as f:
    json.dump(vocab, f)`}
        />

        <CodeBlock
          language="python"
          code={`# Save tokenizer config (from your BPE training)
# This includes merge rules and special token configs
tokenizer.save("custom_tokenizer.json")`}
        />

        <CodeBlock
          language="python"
          code={`# Load into transformers ecosystem
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="custom_tokenizer.json",
    pad_token="[PAD]",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Vocabulary Size Considerations</Title>
        <Text mt="md">
          Custom tokenizer vocabulary size impacts model architecture and performance.
        </Text>

        <CodeBlock
          language="python"
          code={`# Get vocabulary size
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")  # e.g., 32000`}
        />

      </div>

      <div data-slide>
        <Title order={2}>Transformer Package Usage</Title>
        <Text mt="md">
          The Hugging Face transformers library provides model architectures for encoder-only,
          decoder-only, and encoder-decoder configurations.
        </Text>

      </div>

      <div data-slide>
        <Title order={2}>Defining Encoder Architecture</Title>
        <Text mt="md">
          Encoder models process input bidirectionally and output contextualized representations.
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import BertConfig, BertModel

config = BertConfig(
    vocab_size=30000,           # Size of your vocabulary
    hidden_size=768,             # Embedding dimension
    num_hidden_layers=12,        # Number of transformer blocks
    num_attention_heads=12,      # Heads per layer
    intermediate_size=3072,      # FFN hidden dimension
    max_position_embeddings=512  # Max sequence length
)`}
        />

        <CodeBlock
          language="python"
          code={`# Initialize model from config
model = BertModel(config)
print(f"Parameters: {model.num_parameters():,}")
# Parameters: 109,482,240`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Understanding Key Configuration Parameters</Title>

        <Title order={3} mt="lg">Embedding Size (hidden_size)</Title>
        <Text mt="md">
          The embedding dimension determines the vector representation size for each token.
          Each token is mapped to a dense vector of this size.
        </Text>

        <CodeBlock
          language="python"
          code={`# Token embeddings: vocab_size × hidden_size
# For BERT-base: 30,000 × 768 = 23,040,000 parameters

input_ids = torch.tensor([[101, 7592, 2088, 102]])
embeddings = model.embeddings.word_embeddings(input_ids)
print(embeddings.shape)  # (1, 4, 768)`}
        />

        <Text mt="md">
          Larger embedding sizes capture more information but increase computational cost.
          Common sizes: 768 (base), 1024 (large).
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Position Encodings in BERT</Title>
        <Text mt="md">
          BERT uses learned positional embeddings (not sinusoidal like original Transformer).
          Each position has a trainable embedding vector.
        </Text>

        <CodeBlock
          language="python"
          code={`# Position embeddings: max_position_embeddings × hidden_size
# For BERT-base: 512 × 768 = 393,216 parameters

position_embeddings = model.embeddings.position_embeddings
print(position_embeddings.weight.shape)  # (512, 768)`}
        />

        <Text mt="md">Final token representation combines three embeddings:</Text>
        <CodeBlock
          language="python"
          code={`# Token embedding + Position embedding + Segment embedding
# All of dimension hidden_size (768)
final_embedding = (
    word_embedding[token_id] +
    position_embedding[position] +
    token_type_embedding[segment_id]
)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Defining Decoder Architecture</Title>
        <Text mt="md">
          Decoder models process input autoregressively with causal masking for text generation.
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768
)`}
        />

        <CodeBlock
          language="python"
          code={`    n_layer=12,
    n_head=12,
    n_inner=3072,
    activation_function='gelu_new'
)`}
        />

        <CodeBlock
          language="python"
          code={`# Initialize model with language modeling head
model = GPT2LMHeadModel(config)
print(f"Parameters: {model.num_parameters():,}")`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Encoder Training: Data Preparation</Title>
        <Text mt="md">
          Encoder models for classification require labeled data with proper tokenization
          and special token management.
        </Text>

        <Title order={3} mt="lg">Dataset Structure</Title>
        <CodeBlock
          language="python"
          code={`from torch.utils.data import Dataset

class EncoderDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,      # Cut sequences longer than max_length
            padding='max_length', # Pad all to max_length
            max_length=max_length, # Maximum sequence length
            return_tensors='pt'   # Return PyTorch tensors
        )
        self.labels = labels`}
        />

        <CodeBlock
          language="python"
          code={`    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Understanding Tokenization Parameters</Title>

        <Title order={3} mt="lg">truncation=True</Title>
        <Text mt="md">
          Sequences longer than max_length are truncated to fit model constraints.
        </Text>
        <CodeBlock
          language="python"
          code={`# Without truncation: may exceed model's max_position_embeddings
text = "Very long text..." * 100  # 500+ tokens
tokens = tokenizer.encode(text, max_length=128, truncation=True)
print(len(tokens))  # 128 (truncated from 500+)`}
        />

        <Title order={3} mt="lg">padding='max_length'</Title>
        <Text mt="md">
          All sequences are padded to the same length for efficient batching.
          This ensures uniform tensor shapes in batches.
        </Text>
        <CodeBlock
          language="python"
          code={`# With padding='max_length': all sequences same size
texts = ["Hello", "Hello world how are you"]
encoded = tokenizer(texts, padding='max_length', max_length=10)
print([len(ids) for ids in encoded['input_ids']])
# [10, 10] - both padded to max_length`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Padding Strategies for Encoders</Title>
        <Text mt="md">
          Different padding strategies trade off efficiency and memory usage.
        </Text>

        <CodeBlock
          language="python"
          code={`# Strategy 1: Pad to max_length (used in dataset creation)
# - Fixed size for all batches
# - More memory but simpler code
encoded = tokenizer(texts, padding='max_length', max_length=128)`}
        />

        <CodeBlock
          language="python"
          code={`# Strategy 2: Pad to longest in batch (used in DataLoader)
# - Variable batch size, less memory waste
# - Requires custom collate function
from transformers import DataCollatorWithPadding

collator = DataCollatorWithPadding(tokenizer=tokenizer)`}
        />

        <Text mt="md">
          For encoders, padding is essential because all sequences in a batch must have
          the same length for matrix operations.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Encoder Training: Special Tokens</Title>
        <Text mt="md">
          Encoder models use specific special tokens to mark sequence boundaries and enable
          various tasks.
        </Text>

        <Table striped mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Token</Table.Th>
              <Table.Th>Purpose</Table.Th>
              <Table.Th>Position</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>[CLS]</Table.Td>
              <Table.Td>Classification embedding</Table.Td>
              <Table.Td>Start of sequence</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>[SEP]</Table.Td>
              <Table.Td>Separator between segments</Table.Td>
              <Table.Td>End of each segment</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>[PAD]</Table.Td>
              <Table.Td>Padding to max length</Table.Td>
              <Table.Td>After [SEP] token</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>[MASK]</Table.Td>
              <Table.Td>Masked language modeling</Table.Td>
              <Table.Td>Random positions</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Text mt="md">Example token sequence structure:</Text>
        <CodeBlock
          language="python"
          code={`# Input: "Hello world"
# Tokens: [CLS] hello world [SEP] [PAD] [PAD] ...
# IDs:    [101,  7592, 2088, 102,  0,    0,   ...]`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Encoder Training: Attention Masks</Title>
        <Text mt="md">
          Attention masks prevent the model from attending to padding tokens, ensuring correct
          computation and gradient flow.
        </Text>

        <BlockMath>{'\\text{mask}[i] = \\begin{cases} 1 & \\text{if real token} \\\\ 0 & \\text{if [PAD]} \\end{cases}'}</BlockMath>

        <CodeBlock
          language="python"
          code={`# Tokenized with padding
encoded = tokenizer(
    ["Hello world", "Short"],
    padding='max_length',
    max_length=8
)`}
        />

        <CodeBlock
          language="python"
          code={`# Attention masks automatically created
print(encoded['attention_mask'])
# [[1, 1, 1, 1, 0, 0, 0, 0],
#  [1, 1, 1, 0, 0, 0, 0, 0]]`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Encoder Training: Masked Language Modeling (MLM)</Title>
        <Text mt="md">
          For pre-training encoders like BERT, we randomly mask tokens and train the model
          to predict them. This happens during data preparation, not in the model.
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import DataCollatorForLanguageModeling

# Create data collator that masks 15% of tokens randomly
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)`}
        />

        <Text mt="md">The collator randomly masks tokens during training:</Text>
        <CodeBlock
          language="python"
          code={`# Original: "The cat sat on the mat"
# Masked:   "The [MASK] sat [MASK] the mat"
# Model predicts: "cat" and "on"

# During each epoch, different tokens are masked (randomized)`}
        />

        <Text mt="md">
          The attention_mask is different from MLM masking - it marks padding positions,
          not masked tokens. Both real and [MASK] tokens have attention_mask=1.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>How DataCollatorForLanguageModeling Works</Title>
        <Text mt="md">
          The data collator creates both masked input_ids and labels with -100 values.
          This controls which positions contribute to the loss.
        </Text>

        <Title order={3} mt="lg">Step-by-Step Process</Title>
        <CodeBlock
          language="python"
          code={`# Step 1: Original sequence
input_ids = [101, 4937, 2000, 2006, 1996, 4500, 102]
#            [CLS] cat   sat  on   the  mat  [SEP]

# Step 2: Randomly select ~15% positions to mask (e.g., positions 1 and 3)
# Step 3: Create labels (copy of input_ids)
labels = [101, 4937, 2000, 2006, 1996, 4500, 102]`}
        />

        <CodeBlock
          language="python"
          code={`# Step 4: Set labels to -100 for NON-masked positions
labels = [-100, 4937, -100, 2006, -100, -100, -100]
#         ^^^^  ^^^^  ^^^^  ^^^^  ^^^^  ^^^^  ^^^^
#         skip  PRED  skip  PRED  skip  skip  skip

# Step 5: Replace masked positions in input_ids with [MASK]
input_ids = [101, 103, 2000, 103, 1996, 4500, 102]
#            [CLS] [MASK] sat [MASK] the  mat [SEP]`}
        />

        <Text mt="md">
          The batch returned contains both input_ids (with [MASK] tokens) and labels
          (with -100 for non-masked positions).
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Understanding the -100 Label Mechanism</Title>
        <Text mt="md">
          PyTorch's CrossEntropyLoss ignores positions where labels = -100.
          This is how we control which positions contribute to the loss.
        </Text>

        <CodeBlock
          language="python"
          code={`# After data_collator processes the batch:
input_ids = [101, 103, 2000, 103, 1996, 4500, 102, 0, 0]
#            [CLS] [MASK] sat [MASK] the  mat [SEP] [PAD] [PAD]

labels = [-100, 4937, -100, 2006, -100, -100, -100, -100, -100]
#        ^^^^  ^^^^  ^^^^  ^^^^  ^^^^  ^^^^  ^^^^  ^^^^  ^^^^
#        skip   cat  skip   on   skip  skip  skip  skip  skip`}
        />

        <Text mt="md">Loss computation:</Text>
        <CodeBlock
          language="python"
          code={`# Model outputs predictions for ALL positions
outputs = model(input_ids=input_ids, labels=labels)

# But CrossEntropyLoss only computes loss where labels != -100
# Loss computed on: position 1 (cat) and position 3 (on)
# Loss ignored on: [CLS], unmasked tokens, [SEP], [PAD]
loss = outputs.loss  # Only from masked positions`}
        />

        <Text mt="md">
          This ensures the model learns to predict only the masked tokens,
          not all tokens or padding tokens.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Encoder Training: Batch Processing</Title>
        <Text mt="md">
          Efficient training requires proper batching with consistent sequence lengths.
        </Text>

        <CodeBlock
          language="python"
          code={`from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=data_collator  # Applies MLM masking
)`}
        />

        <Text mt="md">Processing a batch:</Text>
        <CodeBlock
          language="python"
          code={`for batch in dataloader:
    input_ids = batch['input_ids']        # (32, 128)
    attention_mask = batch['attention_mask']  # (32, 128)
    labels = batch['labels']              # (32, 128) with -100

    # data_collator created labels with -100 for non-masked positions
    # attention_mask: 1 for real tokens (including [MASK]), 0 for [PAD]
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss  # Only computed where labels != -100`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Encoder Training: Loss and Optimization</Title>
        <Text mt="md">
          Masked language modeling loss is computed only on masked positions.
          We use BertForMaskedLM with our custom config for pre-training.
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import BertForMaskedLM

# Use our custom config from earlier
model = BertForMaskedLM(config)
# config had: vocab_size=30000, hidden_size=768, etc.`}
        />

        <Text mt="md">Loss computation is handled internally:</Text>
        <CodeBlock
          language="python"
          code={`outputs = model(
    input_ids=input_ids,        # Contains [MASK] tokens
    attention_mask=attention_mask,
    labels=labels               # Token IDs with -100 for non-masked
)
loss = outputs.loss  # Cross-entropy only on masked positions
logits = outputs.logits  # (batch_size, seq_len, vocab_size)`}
        />

        <Text mt="md">
          The model outputs predictions for all positions, but CrossEntropyLoss only
          computes loss where labels are not -100 (i.e., on masked positions).
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Encoder Training: Complete Example</Title>
        <Text mt="md">
          Full training loop for masked language modeling using DataCollatorForLanguageModeling.
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()`}
        />

        <CodeBlock
          language="python"
          code={`for epoch in range(3):
    for batch in dataloader:
        # data_collator already prepared:
        # - input_ids (with [MASK] tokens)
        # - labels (with -100 for non-masked positions)
        # - attention_mask (1 for real tokens, 0 for padding)

        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )`}
        />

        <CodeBlock
          language="python"
          code={`        loss = outputs.loss  # Only computed where labels != -100
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Encoder Inference: Predicting Masked Tokens</Title>
        <Text mt="md">
          After training, the model can predict masked tokens. Typically, encoder models
          are then fine-tuned for downstream tasks like classification or question answering.
        </Text>

        <CodeBlock
          language="python"
          code={`model.eval()
import torch

# Input with masked token
text = "The cat sat on the [MASK]"
inputs = tokenizer(text, return_tensors='pt')`}
        />

        <CodeBlock
          language="python"
          code={`with torch.no_grad():
    outputs = model(**inputs)

# Get predictions for all positions
logits = outputs.logits  # (batch_size, seq_len, vocab_size)

# Find masked position and get top prediction
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero()
predicted_token_id = logits[0, mask_token_index, :].argmax(dim=-1)`}
        />

        <CodeBlock
          language="python"
          code={`predicted_token = tokenizer.decode(predicted_token_id)
print(f"Predicted: {predicted_token}")  # "mat" or "floor"`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Decoder Training: Data Preparation</Title>
        <Text mt="md">
          Decoder models for language modeling require causal data structure where each token
          predicts the next. Unlike encoders, decoders often use minimal padding.
        </Text>

        <CodeBlock
          language="python"
          code={`class DecoderDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(
                text,
                max_length=max_length,
                truncation=True
                # No padding='max_length' here!
            )
            self.examples.append(torch.tensor(tokens))`}
        />

        <CodeBlock
          language="python"
          code={`    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Why Decoders Need Less Padding</Title>
        <Text mt="md">
          Decoder training differs from encoder training in padding requirements.
        </Text>

        <Title order={3} mt="lg">Encoder Approach</Title>
        <CodeBlock
          language="python"
          code={`# Encoders: Pre-pad all sequences in dataset
# Every sequence is max_length (e.g., 512)
# Fixed memory, but wasteful for short sequences
encodings = tokenizer(texts, padding='max_length', max_length=512)`}
        />

        <Title order={3} mt="lg">Decoder Approach</Title>
        <CodeBlock
          language="python"
          code={`# Decoders: Store variable-length sequences
# Pad only when creating batches (dynamic padding)
# More memory efficient, especially for varied lengths
tokens = tokenizer.encode(text, truncation=True)  # No padding yet!`}
        />

        <Text mt="md">
          Padding is applied dynamically during batching using a collate function,
          reducing memory waste.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Decoder Training: Special Tokens</Title>
        <Text mt="md">
          Decoder models use minimal special tokens for sequence boundaries and generation control.
        </Text>

        <Table striped mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Token</Table.Th>
              <Table.Th>Purpose</Table.Th>
              <Table.Th>Usage</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>&lt;BOS&gt;</Table.Td>
              <Table.Td>Beginning of sequence</Table.Td>
              <Table.Td>Start generation</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>&lt;EOS&gt;</Table.Td>
              <Table.Td>End of sequence</Table.Td>
              <Table.Td>Stop generation</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>&lt;PAD&gt;</Table.Td>
              <Table.Td>Padding</Table.Td>
              <Table.Td>Batch uniformity</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Text mt="md">Configure for GPT-style models:</Text>
        <CodeBlock
          language="python"
          code={`from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Decoder Training: Causal Masking</Title>
        <Text mt="md">
          Causal attention masks ensure each position only attends to previous positions,
          enabling autoregressive generation.
        </Text>

        <BlockMath>{'\\text{mask}[i,j] = \\begin{cases} 0 & \\text{if } j \\leq i \\\\ -\\infty & \\text{if } j > i \\end{cases}'}</BlockMath>

        <Text mt="md">Attention mechanism automatically applies causal mask:</Text>
        <CodeBlock
          language="python"
          code={`# Input: "The cat sat"
# Position 0 (The) sees: [The]
# Position 1 (cat) sees: [The, cat]
# Position 2 (sat) sees: [The, cat, sat]`}
        />

        <Text mt="md">
          The model is trained to predict the next token at each position.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Decoder Training: Data Collator</Title>
        <Text mt="md">
          For decoder models, we use the same DataCollatorForLanguageModeling but with
          mlm=False to enable causal language modeling instead of masked language modeling.
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import DataCollatorForLanguageModeling

# Create data collator for causal LM (decoder)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not Masked LM
)`}
        />

        <Text mt="md">
          The data collator automatically handles padding, label shifting, and -100 masking
          for next-token prediction during batch creation.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>How DataCollatorForLanguageModeling Works (mlm=False)</Title>
        <Text mt="md">
          When mlm=False, the collator prepares data for causal language modeling by
          shifting labels for next-token prediction.
        </Text>

        <Title order={3} mt="lg">Step-by-Step Process</Title>
        <CodeBlock
          language="python"
          code={`# Step 1: Batch of variable-length sequences
# Sequence 1: [1, 2, 3, 4, 5]
# Sequence 2: [6, 7, 8]

# Step 2: Pad to longest in batch
input_ids = [[1, 2, 3, 4, 5],
             [6, 7, 8, 0, 0]]  # 0 = pad_token_id`}
        />

        <CodeBlock
          language="python"
          code={`# Step 3: Create labels by cloning input_ids
labels = input_ids.clone()

# Step 4: Shift labels left for next-token prediction
# labels[:, :-1] = input_ids[:, 1:]
# labels[:, -1] = -100
labels = [[2, 3, 4, 5, -100],
          [7, 8, 0, 0, -100]]`}
        />

        <CodeBlock
          language="python"
          code={`# Step 5: Mask padding positions in labels
# labels[labels == pad_token_id] = -100
labels = [[2, 3, 4, 5, -100],
          [7, 8, -100, -100, -100]]
#              ^^^^  ^^^^  ^^^^
#              real  pad   pad   last_pos`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Understanding Decoder Label Shifting</Title>
        <Text mt="md">
          The collator shifts labels so each position predicts the next token.
        </Text>

        <CodeBlock
          language="python"
          code={`# Example: "The cat sat"
input_ids = [[101, 202, 303]]

# After collator processing:
# input_ids stays: [101, 202, 303]
# labels become:   [202, 303, -100]

# Training objective:
# Position 0: sees 101 (The) → predicts 202 (cat)
# Position 1: sees 202 (cat) → predicts 303 (sat)
# Position 2: sees 303 (sat) → predicts -100 (ignored)`}
        />

        <Text mt="md">
          The -100 at the last position is necessary because there is no next token
          to predict after the sequence ends.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Understanding the -100 Label Values</Title>
        <Text mt="md">
          For decoders, -100 appears in labels for two reasons, both handled by the collator.
        </Text>

        <CodeBlock
          language="python"
          code={`# Example with padding:
input_ids = [[1, 2, 3, 4, 0, 0]]  # 0 = pad_token_id

# After DataCollatorForLanguageModeling(mlm=False):
labels = [[2, 3, 4, -100, -100, -100]]
#                   ^^^^  ^^^^  ^^^^`}
        />

        <List spacing="sm" mt="md">
          <List.Item>
            <strong>Last real position:</strong> Token 4 has no next token to predict,
            so after shifting, this position gets -100
          </List.Item>
          <List.Item>
            <strong>Padding positions:</strong> Padding tokens (0) are replaced with
            -100 so loss is not computed on padding
          </List.Item>
        </List>

        <Text mt="md">
          CrossEntropyLoss ignores all positions where labels = -100, ensuring loss is
          computed only on valid next-token predictions.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Decoder Training: Batch Processing</Title>
        <Text mt="md">
          Use the data collator in DataLoader to automatically prepare batches with
          proper padding, shifting, and label masking.
        </Text>

        <CodeBlock
          language="python"
          code={`from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator  # Handles padding, shifting, -100
)`}
        />

        <Text mt="md">Processing batches:</Text>
        <CodeBlock
          language="python"
          code={`for batch in dataloader:
    input_ids = batch['input_ids']        # Original tokens
    attention_mask = batch['attention_mask']  # 1=real, 0=pad
    labels = batch['labels']              # Shifted with -100

    # labels has -100 for: padding positions and last position
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss  # Only computed where labels != -100`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Training vs Inference: Key Difference</Title>
        <Text mt="md">
          Decoder models behave differently during training and inference.
        </Text>

        <Title order={3} mt="lg">Training Mode (Teacher Forcing)</Title>
        <CodeBlock
          language="python"
          code={`# Training: Parallel processing with ground truth
# Input:  [The, cat, sat, on]
# Label:  [cat, sat, on, mat]
# All positions computed in one forward pass
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss  # Single backward pass`}
        />

        <Title order={3} mt="lg">Inference Mode (Autoregressive)</Title>
        <CodeBlock
          language="python"
          code={`# Inference: Sequential generation
# Step 1: [The] → predict "cat"
# Step 2: [The, cat] → predict "sat"
# Step 3: [The, cat, sat] → predict "on"
# Requires multiple forward passes (slow!)`}
        />

        <Text mt="md">
          Teacher forcing during training is much faster than inference because
          all predictions are computed in parallel.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Decoder Training: Complete Example</Title>
        <Text mt="md">
          Full training loop using DataCollatorForLanguageModeling for automatic
          label preparation. We use the GPT2 model we defined earlier with custom config.
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import AdamW

# Use our custom config from earlier
# config had: vocab_size=50257, n_embd=768, n_layer=12, etc.
model = GPT2LMHeadModel(config)
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()`}
        />

        <CodeBlock
          language="python"
          code={`for epoch in range(3):
    for batch in dataloader:
        # data_collator already prepared:
        # - input_ids (padded sequences)
        # - labels (shifted with -100 for padding and last pos)
        # - attention_mask (1 for real tokens, 0 for padding)

        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )`}
        />

        <CodeBlock
          language="python"
          code={`        loss = outputs.loss  # Only computed where labels != -100
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Decoder Inference: Greedy Generation</Title>
        <Text mt="md">
          Generate text by iteratively predicting the most likely next token.
        </Text>

        <CodeBlock
          language="python"
          code={`model.eval()
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')`}
        />

        <CodeBlock
          language="python"
          code={`for _ in range(50):  # Generate 50 tokens
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)`}
        />

        <CodeBlock
          language="python"
          code={`    input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    if next_token == tokenizer.eos_token_id:
        break

generated_text = tokenizer.decode(input_ids[0])`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Decoder Inference: Sampling Strategies</Title>
        <Text mt="md">
          Control generation diversity through temperature and top-k/top-p sampling.
        </Text>

        <CodeBlock
          language="python"
          code={`# Temperature sampling
temperature = 0.8
next_token_logits = outputs.logits[:, -1, :] / temperature
probs = torch.softmax(next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)`}
        />

        <Text mt="md">Using generate API with parameters:</Text>
        <CodeBlock
          language="python"
          code={`generated = model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>DataCollatorForLanguageModeling: Encoder vs Decoder</Title>
        <Text mt="md">
          The same DataCollator class handles both encoder and decoder training,
          controlled by the mlm parameter.
        </Text>

        <Title order={3} mt="lg">Encoder (mlm=True)</Title>
        <CodeBlock
          language="python"
          code={`# For BERT-style masked language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)
# Creates: Random masking, labels with -100 for non-masked positions`}
        />

        <Title order={3} mt="lg">Decoder (mlm=False)</Title>
        <CodeBlock
          language="python"
          code={`# For GPT-style causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
# Creates: Label shifting, labels with -100 for padding and last position`}
        />

        <Text mt="md">
          Both use -100 in labels to control which positions contribute to loss computation.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Key Training Parameters Summary</Title>

        <Title order={3} mt="lg">Encoder Models (BERT-style)</Title>
        <List spacing="xs" mt="sm">
          <List.Item>Data: DataCollatorForLanguageModeling with mlm=True</List.Item>
          <List.Item>Special tokens: [CLS], [SEP], [PAD], [MASK]</List.Item>
          <List.Item>Attention: Bidirectional with attention masks</List.Item>
          <List.Item>Loss: Computed only on masked positions (~15%)</List.Item>
          <List.Item>Inference: Single forward pass per sequence</List.Item>
        </List>

        <Title order={3} mt="lg">Decoder Models (GPT-style)</Title>
        <List spacing="xs" mt="sm">
          <List.Item>Data: DataCollatorForLanguageModeling with mlm=False</List.Item>
          <List.Item>Special tokens: &lt;BOS&gt;, &lt;EOS&gt;, &lt;PAD&gt;</List.Item>
          <List.Item>Attention: Causal masking for autoregressive generation</List.Item>
          <List.Item>Loss: Computed on all real next-token predictions</List.Item>
          <List.Item>Inference: Iterative generation with sampling</List.Item>
        </List>

        <Text mt="md">
          Both use -100 in labels to mark positions ignored by CrossEntropyLoss.
        </Text>
      </div>

    </div>
  );
}
