# Module 8: Natural Language Processing - Exercises Overview

This document outlines the planned exercises for Module 8, covering fundamental to advanced concepts in Natural Language Processing with Transformers.

## Exercise Structure

Module 8 includes **5 exercises** progressing from basic tokenization to advanced agentic systems:

---

## Exercise 0: Tokenization Fundamentals
**Status:** ✅ Implemented
**Type:** Warmup Exercise

### Objectives
- Understand text tokenization methods and algorithms
- Implement Byte-Pair Encoding (BPE) tokenization
- Train a custom BPE tokenizer from scratch
- Explore tokenization impact on downstream tasks

### Key Topics
- Word-level vs. subword-level tokenization
- Common algorithms: BPE, WordPiece, SentencePiece
- Handling special tokens and out-of-vocabulary words
- Token vocabularies and frequencies
- Preprocessing text for language models

### Deliverables
- Custom BPE tokenizer implementation
- Comparative analysis of different tokenization strategies
- Token distribution analysis on sample datasets

---

## Exercise 1: Transformer Architecture and Training ⭐
**Status:** ✅ Implemented
**Type:** Marked Exercise (Graded)

### Objectives
- Build a complete Transformer model from scratch
- Understand self-attention mechanisms
- Train a transformer for sequence-to-sequence tasks
- Apply the model to machine translation

### Key Topics
- Self-attention and multi-head attention mechanisms
- Positional encodings and embedding layers
- Feed-forward networks and layer normalization
- Encoder-decoder architecture
- Tokenization strategies for translation tasks
- Beam search for sequence generation
- BLEU score evaluation

### Deliverables
- Complete Transformer implementation in PyTorch
- Trained English-to-German translation model
- Performance evaluation with BLEU metrics
- Analysis of attention patterns

---

## Exercise 2: Attention Mechanisms and Zero-Shot Learning
**Status:** ✅ Implemented (partial - needs attention visualization)
**Type:** Practice Exercise

### Objectives
- Deep dive into attention mechanisms
- Visualize and interpret attention patterns
- Leverage pre-trained models for zero-shot tasks
- Understand and extract model embeddings

### Key Topics
#### Part A: Attention Mechanisms
- Scaled dot-product attention
- Multi-head attention visualization
- Attention weight interpretation
- Cross-attention vs. self-attention

#### Part B: Zero-Shot Learning
- Zero-shot classification without fine-tuning
- Zero-shot question answering
- Zero-shot sentiment analysis
- Zero-shot translation and summarization
- Mathematical reasoning capabilities

#### Part C: Embeddings from Hugging Face Models
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

## Exercise 3: Fine-Tuning with LoRA ⭐
**Status:** ✅ Implemented
**Type:** Marked Exercise (Graded)

### Objectives
- Master parameter-efficient fine-tuning techniques
- Implement LoRA (Low-Rank Adaptation) fine-tuning
- Fine-tune large language models for specific tasks
- Optimize memory and computational efficiency

### Key Topics
- Parameter-efficient fine-tuning principles
- LoRA architecture and rank dimensionality
- PEFT (Parameter-Efficient Fine-Tuning) library
- Dataset preparation for fine-tuning (SST-2)
- Hyperparameter optimization (rank, alpha, dropout)
- Training loop implementation
- Model evaluation and deployment
- Merging and exporting LoRA weights

### Deliverables
- LoRA fine-tuned sentiment classification model
- Performance comparison: full fine-tuning vs. LoRA
- Memory and computational cost analysis
- Deployment-ready model with merged weights

---

## Exercise 4: Agentic Systems with LangGraph
**Status:** 🔨 Planned
**Type:** Practice Exercise

### Objectives
- Understand agentic AI architectures
- Build multi-agent systems with LangGraph
- Implement tool use and reasoning loops
- Create autonomous task-solving agents

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

#### Part C: Building Practical Agents
- Research agent with web search capabilities
- Code generation and debugging agent
- Multi-agent collaboration systems
- Error handling and retry mechanisms

#### Part D: Advanced Patterns
- Human-in-the-loop workflows
- Agent memory and context persistence
- Streaming and real-time responses
- Production deployment considerations

### Deliverables
- Multi-agent research assistant
- Code analysis and generation agent
- Collaborative agent system
- Performance and cost analysis

---

## Exercise Progression Map

```
Exercise 0: Tokenization
    ↓
Exercise 1: Transformer Architecture ⭐
    ↓
Exercise 2: Attention + Zero-Shot + Embeddings
    ↓
Exercise 3: LoRA Fine-Tuning ⭐
    ↓
Exercise 4: Agentic Systems
```

## Grading Information

**Marked Exercises (30% each of module grade):**
- Exercise 1: Transformer Architecture and Training
- Exercise 3: LoRA Fine-Tuning

**Practice Exercises:**
- Exercise 0: Tokenization Fundamentals
- Exercise 2: Attention and Zero-Shot Learning
- Exercise 4: Agentic Systems

## Prerequisites

### Required Knowledge
- Python programming proficiency
- Basic understanding of neural networks
- Familiarity with PyTorch or similar frameworks
- Linear algebra fundamentals
- Basic probability and statistics

### Technical Setup
- Python 3.8+
- PyTorch 2.0+
- Transformers library (Hugging Face)
- PEFT library
- LangGraph
- GPU access recommended (Colab/local)

## Learning Outcomes

By completing these exercises, students will be able to:

1. **Tokenization & Preprocessing**
   - Implement and train custom tokenizers
   - Choose appropriate tokenization strategies

2. **Transformer Architecture**
   - Build transformers from scratch
   - Understand attention mechanisms deeply
   - Train models for sequence tasks

3. **Pre-trained Models**
   - Leverage Hugging Face ecosystem
   - Apply zero-shot learning techniques
   - Extract and utilize embeddings

4. **Efficient Fine-Tuning**
   - Implement LoRA and other PEFT methods
   - Optimize computational resources
   - Deploy fine-tuned models

5. **Agentic AI**
   - Design multi-agent systems
   - Implement tool-using agents
   - Build production-ready AI applications

## Resources

### Primary Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

### Recommended Reading
- "Attention Is All You Need" (Vaswani et al., 2017)
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "Chain-of-Thought Prompting" (Wei et al., 2022)

### Datasets
- WikiText-103 (tokenization)
- Multi30k (English-German translation)
- SST-2 (sentiment classification)
- Custom datasets for agentic applications

---

## Notes

- All exercises include Jupyter notebooks with starter code
- Solutions and detailed explanations provided after submission deadlines
- Office hours available for questions on marked exercises
- Collaboration encouraged on practice exercises
- Code must be original for marked exercises

**Last Updated:** 2025-10-17