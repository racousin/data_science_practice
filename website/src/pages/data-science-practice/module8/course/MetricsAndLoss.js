import React from "react";
import { Text, Title, List, Table, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";

const MetricsAndLoss = () => {
  return (
    <>
      <div data-slide>
        <Title order={1}>Metrics and Loss Functions in NLP</Title>

        <Text mt="md">
          NLP tasks require specialized loss functions and evaluation metrics that account for the sequential
          and discrete nature of language.
        </Text>

        <Text mt="md">
          Loss functions guide model training through gradient descent, while metrics evaluate performance
          on validation and test data.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Cross-Entropy Loss</Title>

        <Text mt="md">
          Cross-entropy is the fundamental loss function for most NLP tasks involving classification or
          next-token prediction.
        </Text>

        <Title order={3} mt="lg">Single Token Classification</Title>
        <Text mt="sm">
          For a single prediction with <InlineMath math="K" /> classes:
        </Text>
        <BlockMath math="L_{CE} = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)" />

        <Text mt="md">
          Where:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath math="y_k" />: True probability of class k (1 for true class, 0 otherwise)</List.Item>
          <List.Item><InlineMath math="\hat{y}_k" />: Predicted probability of class k</List.Item>
          <List.Item><InlineMath math="\hat{y} \in \mathbb{R}^K" />: Output from softmax layer</List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/cross-entropy-visualization.png"
            alt="Visualization of cross-entropy loss computation with probability distributions"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Cross-entropy loss: measuring divergence between predicted and true distributions
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Sequence Cross-Entropy Loss</Title>

        <Text mt="md">
          For sequence prediction (language modeling, machine translation), we compute the average
          cross-entropy across all time steps:
        </Text>

        <BlockMath math="L = -\frac{1}{T}\sum_{t=1}^{T}\sum_{k=1}^{K} y_k^{(t)} \log(\hat{y}_k^{(t)})" />

        <Text mt="lg">
          <strong>Input shape:</strong> Logits <InlineMath math="\in \mathbb{R}^{T \times K}" /> (sequence length T, vocabulary size K)
        </Text>
        <Text mt="sm">
          <strong>Target shape:</strong> Token IDs <InlineMath math="\in \mathbb{Z}^{T}" /> (ground truth tokens)
        </Text>
        <Text mt="sm">
          <strong>Output:</strong> Scalar loss value
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Cross-Entropy Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn.functional as F

# Example: Language modeling with vocabulary size 50257
T, K = 10, 50257  # sequence length, vocabulary size
logits = torch.randn(T, K)  # Model predictions (unnormalized)
targets = torch.randint(0, K, (T,))  # Ground truth token IDs`}
        />

        <CodeBlock
          language="python"
          code={`# Compute cross-entropy loss
loss = F.cross_entropy(
    logits.view(-1, K),  # Reshape to (T, K)
    targets.view(-1)      # Reshape to (T,)
)
print(f"Loss: {loss.item():.4f}")
# Output: Loss: 10.8254 (high loss indicates poor predictions)

# With properly trained model, loss should be much lower (e.g., 2-4)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Perplexity</Title>

        <Text mt="md">
          Perplexity is the exponential of cross-entropy loss, measuring how well a probability model
          predicts a sample.
        </Text>

        <BlockMath math="PPL = \exp(L_{CE}) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(w_t | w_{< t})\right)" />

        <Text mt="lg">
          <strong>Interpretation:</strong> Perplexity represents the average number of choices the model
          is uncertain about at each step.
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>Lower perplexity = better model (more confident predictions)</List.Item>
          <List.Item>Random baseline: perplexity ≈ vocabulary size</List.Item>
          <List.Item>GPT-3: perplexity ≈ 20 on web text</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# Calculate perplexity from loss
perplexity = torch.exp(loss)
print(f"Perplexity: {perplexity.item():.2f}")
# Output: Perplexity: 50126.84`}
        />

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/perplexity-comparison.png"
            alt="Perplexity values comparison across different language models"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Perplexity benchmarks: lower values indicate better language models
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Classification Metrics</Title>

        <Text mt="md">
          For text classification tasks (sentiment analysis, topic classification, NER):
        </Text>

        <Title order={3} mt="lg">Accuracy</Title>
        <BlockMath math="\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}" />

        <Title order={3} mt="lg">Precision</Title>
        <BlockMath math="\text{Precision} = \frac{TP}{TP + FP}" />

        <Title order={3} mt="lg">Recall</Title>
        <BlockMath math="\text{Recall} = \frac{TP}{TP + FN}" />

        <Title order={3} mt="lg">F1 Score</Title>
        <BlockMath math="F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}" />

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/confusion-matrix-metrics.png"
            alt="Confusion matrix showing TP, FP, TN, FN and derived metrics"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Classification metrics derived from confusion matrix: precision, recall, and F1
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Classification Metrics Example</Title>

        <CodeBlock
          language="python"
          code={`from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Example predictions for sentiment classification (0=negative, 1=positive)
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Output: Accuracy: 0.800`}
        />

        <CodeBlock
          language="python"
          code={`# Calculate precision, recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='binary'
)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")
# Output: Precision: 0.833, Recall: 0.833, F1: 0.833`}
        />
      </div>

      <div data-slide>
        <Title order={2}>BLEU Score</Title>

        <Text mt="md">
          BLEU (Bilingual Evaluation Understudy) measures the quality of machine-generated text
          by comparing n-gram overlap with reference translations.
        </Text>

        <BlockMath math="\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)" />

        <Text mt="lg">
          Where:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath math="p_n" />: Precision of n-grams (typically n=1,2,3,4)</List.Item>
          <List.Item><InlineMath math="w_n" />: Weights (usually uniform: 1/N)</List.Item>
          <List.Item><InlineMath math="BP" />: Brevity penalty to penalize short translations</List.Item>
        </List>

        <Text mt="lg">
          <strong>Range:</strong> 0 to 1 (or 0 to 100 when scaled)
        </Text>

        <Text mt="sm" size="sm" fs="italic">
          Reference: Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (2002) - https://aclanthology.org/P02-1040.pdf
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/bleu-ngram-overlap.png"
            alt="Visualization of n-gram overlap between reference and candidate translations"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            BLEU score calculation: n-gram precision with brevity penalty
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>BLEU Score Example</Title>

        <CodeBlock
          language="python"
          code={`from nltk.translate.bleu_score import sentence_bleu

# Reference translations (can have multiple references)
reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]

# Candidate translation
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']

# Calculate BLEU score
score = sentence_bleu(reference, candidate)
print(f"BLEU score: {score:.4f}")
# Output: BLEU score: 1.0000 (perfect match)`}
        />

        <CodeBlock
          language="python"
          code={`# With a less perfect candidate
candidate2 = ['the', 'cat', 'sits', 'on', 'the', 'mat']
score2 = sentence_bleu(reference, candidate2)
print(f"BLEU score: {score2:.4f}")
# Output: BLEU score: 0.7071 (good but not perfect)

# Poor candidate
candidate3 = ['a', 'dog', 'runs', 'in', 'park']
score3 = sentence_bleu(reference, candidate3)
print(f"BLEU score: {score3:.4f}")
# Output: BLEU score: 0.0000 (no matching n-grams)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>ROUGE Score</Title>

        <Text mt="md">
          ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is commonly used for
          evaluating text summarization.
        </Text>

        <Title order={3} mt="lg">ROUGE-N (N-gram overlap)</Title>
        <BlockMath math="\text{ROUGE-N} = \frac{\sum_{S \in \text{Refs}} \sum_{n\text{-gram} \in S} \text{Count}_{\text{match}}(n\text{-gram})}{\sum_{S \in \text{Refs}} \sum_{n\text{-gram} \in S} \text{Count}(n\text{-gram})}" />

        <Title order={3} mt="lg">ROUGE-L (Longest Common Subsequence)</Title>
        <BlockMath math="\text{ROUGE-L} = \frac{LCS(\text{ref}, \text{cand})}{|\text{ref}|}" />

        <Text mt="lg">
          ROUGE focuses on recall (capturing reference content), while BLEU focuses on precision
          (avoiding spurious content).
        </Text>

        <Text mt="sm" size="sm" fs="italic">
          Reference: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries" (2004) - https://aclanthology.org/W04-1013.pdf
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/rouge-metrics-comparison.png"
            alt="Comparison of ROUGE-N and ROUGE-L metrics for text summarization"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            ROUGE metrics: recall-oriented evaluation with n-grams and LCS
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>ROUGE Score Example</Title>

        <CodeBlock
          language="python"
          code={`from rouge_score import rouge_scorer

# Initialize scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Reference and candidate summaries
reference = "The cat is on the mat"
candidate = "The cat sits on the mat"`}
        />

        <CodeBlock
          language="python"
          code={`# Calculate ROUGE scores
scores = scorer.score(reference, candidate)

print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

# Output:
# ROUGE-1: 0.9091 (unigram overlap)
# ROUGE-2: 0.8000 (bigram overlap)
# ROUGE-L: 0.9091 (longest common subsequence)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Token-Level Loss: Masked Language Modeling</Title>

        <Text mt="md">
          Used in models like BERT, where some tokens are masked and must be predicted.
        </Text>

        <BlockMath math="L_{MLM} = -\frac{1}{|M|}\sum_{t \in M}\sum_{k=1}^{K} y_k^{(t)} \log(\hat{y}_k^{(t)})" />

        <Text mt="lg">
          Where:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath math="M" />: Set of masked token positions</List.Item>
          <List.Item><InlineMath math="|M|" />: Number of masked tokens (typically 15% of sequence)</List.Item>
        </List>

        <Text mt="lg">
          <strong>Input shape:</strong> Masked sequence <InlineMath math="\in \mathbb{R}^{T \times d}" />
        </Text>
        <Text mt="sm">
          <strong>Target shape:</strong> Original token IDs <InlineMath math="\in \mathbb{Z}^{|M|}" /> (only for masked positions)
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/masked-language-modeling.png"
            alt="Illustration of masked language modeling with token masking and prediction"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Masked language modeling: predicting masked tokens from bidirectional context
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Sequence-to-Sequence Loss</Title>

        <Text mt="md">
          For tasks like translation where input and output sequences differ:
        </Text>

        <BlockMath math="L_{seq2seq} = -\frac{1}{T_{tgt}}\sum_{t=1}^{T_{tgt}}\sum_{k=1}^{K} y_k^{(t)} \log(\hat{y}_k^{(t)})" />

        <Text mt="lg">
          <strong>Input shapes:</strong>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Source sequence: <InlineMath math="\in \mathbb{Z}^{T_{src}}" /></List.Item>
          <List.Item>Target sequence: <InlineMath math="\in \mathbb{Z}^{T_{tgt}}" /></List.Item>
        </List>

        <Text mt="lg">
          <strong>Note:</strong> Loss is computed only on the target sequence, and typically
          ignores padding tokens using an attention mask.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Handling Padding in Loss</Title>

        <Text mt="md">
          Variable-length sequences are padded to uniform length. Loss computation should ignore padding tokens.
        </Text>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn.functional as F

# Sequences with padding (padding_idx=0)
logits = torch.randn(2, 10, 1000)  # (batch_size, seq_len, vocab_size)
targets = torch.tensor([
    [45, 123, 67, 89, 0, 0, 0, 0, 0, 0],  # Sequence 1 (length 4)
    [12, 34, 56, 78, 90, 23, 45, 0, 0, 0]  # Sequence 2 (length 7)
])`}
        />

        <CodeBlock
          language="python"
          code={`# Compute loss with padding ignored
loss = F.cross_entropy(
    logits.view(-1, 1000),
    targets.view(-1),
    ignore_index=0  # Ignore padding tokens
)
print(f"Loss (ignoring padding): {loss.item():.4f}")`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Label Smoothing</Title>

        <Text mt="md">
          Label smoothing prevents overconfident predictions by distributing a small amount of
          probability mass to incorrect classes.
        </Text>

        <BlockMath math="y'_k = (1 - \epsilon) \cdot y_k + \frac{\epsilon}{K}" />

        <Text mt="lg">
          Where:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath math="\epsilon" />: Smoothing parameter (typically 0.1)</List.Item>
          <List.Item><InlineMath math="K" />: Number of classes (vocabulary size)</List.Item>
        </List>

        <Text mt="lg">
          <strong>Effect:</strong> Reduces overfitting and improves generalization, particularly for large vocabularies.
        </Text>

        <Text mt="sm" size="sm" fs="italic">
          Reference: Szegedy et al., "Rethinking the Inception Architecture" (2016) - https://arxiv.org/abs/1512.00567
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/label-smoothing-effect.png"
            alt="Comparison of hard labels vs smoothed labels probability distributions"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Label smoothing: preventing overconfident predictions by distributing probability mass
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Task-Specific Metrics Summary</Title>

        <Table striped mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Task</Table.Th>
              <Table.Th>Primary Loss</Table.Th>
              <Table.Th>Evaluation Metrics</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Language Modeling</Table.Td>
              <Table.Td>Cross-Entropy</Table.Td>
              <Table.Td>Perplexity</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Text Classification</Table.Td>
              <Table.Td>Cross-Entropy</Table.Td>
              <Table.Td>Accuracy, F1, Precision, Recall</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Machine Translation</Table.Td>
              <Table.Td>Cross-Entropy</Table.Td>
              <Table.Td>BLEU, METEOR</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Summarization</Table.Td>
              <Table.Td>Cross-Entropy</Table.Td>
              <Table.Td>ROUGE, BLEU</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Question Answering</Table.Td>
              <Table.Td>Cross-Entropy</Table.Td>
              <Table.Td>Exact Match, F1</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/nlp-metrics-overview.png"
            alt="Overview of NLP metrics and their applications across different tasks"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Task-specific metrics: matching evaluation methods to NLP objectives
          </Text>
        </Flex>
      </div>
    </>
  );
};

export default MetricsAndLoss;
