# Module 8: Natural Language Processing - Exercises Overview

This document outlines the exercises for Module 8, covering fundamental to advanced concepts in Natural Language Processing with Transformers.

## Exercise Structure

Module 8 includes **5 exercises** progressing from basic tokenization to advanced agentic systems. look at the corresponing page course 
  'module8': [
    { to: '/introduction', label: 'Introduction' },
    { to: '/tokenization', label: 'Tokenization' },
    { to: '/metrics-and-loss', label: 'Metrics and Loss' },
    { to: '/recurrent-networks', label: 'Recurrent Networks' },
    { to: '/attention-layer', label: 'Attention Layer' },
    { to: '/transformer', label: 'Transformer' },
    { to: '/training-transformers', label: 'Training Transformers' },
    { to: '/llm-transfer-learning', label: 'LLM and Transfer Learning' },
    { to: '/embeddings', label: 'Embeddings' },
    { to: '/rag', label: 'RAG' },
    { to: '/agentic', label: 'Agentic' }
  ],

  website/src/pages/data-science-practice/module8
---

## Exercise 0: Understanding Corpus, Tokens, and BPE Implementation
**Type:** Guided Exercise

### Objectives
- Understand what a text corpus is and how it's structured
- Learn about tokens and tokenization fundamentals
- Implement Byte-Pair Encoding (BPE) algorithm from scratch
- Train a custom BPE tokenizer on real data
- Analyze tokenization characteristics and vocabulary

### Dataset
- **TinyStories** (`roneneldan/TinyStories`)
  - Collection of short stories generated for language learning
  - Simple vocabulary and sentence structures
  - Ideal for understanding tokenization concepts

### Key Topics
- What is a corpus? Text data collection and structure
- Tokens: characters, words, subwords
- Tokenization strategies: character-level, word-level, subword-level
- BPE algorithm: merge rules and vocabulary construction
- Special tokens and handling unknown words
- Token frequency analysis
- Vocabulary size impact

### Deliverables
- BPE tokenizer implementation from scratch
- Trained tokenizer on TinyStories corpus
- Vocabulary analysis (frequency distributions, token examples)
- Comparison of different vocabulary sizes
- Token statistics and insights

---

## Exercise 1: Building a Small GPT-like Transformer
**Type:** Guided Exercise

### Objectives
- Build a small GPT-style transformer architecture from scratch
- Prepare training data using the BPE tokenizer from Exercise 0
- Train a language model for next-token prediction
- Generate text and evaluate model performance

### Dataset
- **TinyStories** (`roneneldan/TinyStories`)
  - Same dataset as Exercise 0 for consistency
  - Use the BPE tokenizer trained in Exercise 0
  - use transformer library for our tokenizer and model see website/src/pages/data-science-practice/module8/course/TrainingTransformers.js

### Key Topics
- Decoder-only transformer architecture (GPT-style)
- Self-attention and causal masking
- Multi-head attention mechanisms
- Positional encodings (learned)
- Feed-forward networks and layer normalization
- Next-token prediction training objective
- Data preparation and batching
- Training loop implementation
- Text generation strategies
- Perplexity evaluation

### Deliverables
- Complete GPT-like transformer implementation in PyTorch
- Data preprocessing pipeline using Exercise 0 tokenizer
- Trained language model on TinyStories
- Text generation examples
- Training loss curves and perplexity analysis
- Generated story samples and quality assessment

---

## Exercise 2: Zero-Shot Learning
**Type:** Guided Exercise

### Objectives
- Deep dive into Hugging Face ecosystem
- Leverage pre-trained models for zero-shot tasks
- Understand and extract model embeddings

### Key Topics

#### Part A: Zero-Shot Learning
- Zero-shot classification without fine-tuning
- Zero-shot question answering
- Zero-shot sentiment analysis
- Zero-shot translation and summarization
- Mathematical reasoning capabilities

#### Part B: Embeddings
- Loading and using pre-trained models
- Extracting contextual embeddings
- Sentence and token-level representations
- Similarity computation with embeddings
- Dimensionality reduction and visualization

### Deliverables
- Attention visualization dashboard
- Zero-shot task implementations
- Embedding extraction and analysis toolkit
- Comparative study of different pre-trained models

---

## Exercise 3: Mathematical Problem Solving ⭐
**Type:** Marked Exercise (Graded)

### Objectives
- Apply LLMs to mathematical reasoning tasks
- Evaluate model performance on mathematical problems (try dfifferent pretrained model and/or finetuned with lora)

### Dataset
- **Custom Math Dataset**
  - Simple mathematical exercises (arithmetic, algebra, word problems) (I will provide it later as a simple dataset X= probleme Y = value solution (it will be always a number)), as previous exercises, they will be a train and test set (test will be only the question without target)
  - The metics will be mse, provide a pipeline with dummy model to evaluate is bad performance
  - Student chooses approach: prompting or fine-tuning

### Student Choice: 
- test pretraine model from hugging face with different prompt
- id not enough fine tune it

### Deliverables
- ipynb
- submission.csv (prediction on test), if you have no numerical value in target column, your score will be +inf

---

## Exercise 4: Agentic Systems with LangGraph
**Type:** Guided Exercise

### Objectives
- Understand agentic AI architectures
- Build agent systems with LangGraph
- Implement tool use and function calling
- Create reasoning and planning loops

### Model
- **Qwen** (suitable for Colab environment)
  - Efficient model size for free CPU/GPU resources
  - Strong tool-use capabilities
  - Good reasoning performance

### Key Topics

#### Part A: Introduction to Agentic Systems
- Agent architectures and paradigms
- Tool use and function calling
- Reasoning and planning loops
- Memory and state management

#### Part B: LangGraph Fundamentals
- Graph-based agent workflows
- Node and edge definitions
- State management across the graph
- Conditional routing and branching
- Building simple agent architectures

#### Part C: Implementing Tools
- Defining custom tools for agents
- Tool schemas and descriptions
- Tool execution and result handling
- Error handling and retry mechanisms

#### Part D: Building Practical Agents
- Simple task-solving agents
- Multi-step reasoning workflows
- Tool selection and orchestration