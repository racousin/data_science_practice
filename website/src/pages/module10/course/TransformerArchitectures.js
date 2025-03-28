import React from 'react';
import { Title, Text, Stack, Grid, Box, List, Table, Divider, Accordion } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const TransformerArchitectures = () => {
  return (
    <Stack spacing="xl" className="w-full">
      {/* Introduction to LLMs */}
      <Title order={1} id="llm-introduction">Large Language Models (LLMs): An Overview</Title>
      
      <Stack spacing="md">
        <Text>
          Large Language Models (LLMs) are advanced AI systems trained on vast amounts of text data to understand, generate, and manipulate human language. These models leverage transformer architectures to process and generate text with remarkable fluency and contextual understanding. LLMs have evolved from basic transformer implementations to increasingly sophisticated architectures that can perform a wide range of language tasks.
        </Text>

        <Box className="p-4 border rounded">
          <Title order={4}>Key Characteristics of LLMs</Title>
          <List>
            <List.Item><strong>Scale:</strong> Trained on billions to trillions of parameters</List.Item>
            <List.Item><strong>Transfer Learning:</strong> Pre-trained on large corpora and fine-tuned for specific tasks</List.Item>
            <List.Item><strong>Emergent Abilities:</strong> Capabilities that emerge only at larger scales</List.Item>
            <List.Item><strong>Few-shot Learning:</strong> Ability to learn from just a few examples</List.Item>
            <List.Item><strong>In-context Learning:</strong> Learning from examples provided within the prompt</List.Item>
          </List>
        </Box>
        
        <Text>
          While all modern LLMs are based on transformer architecture, they differ significantly in their design choices, training objectives, and target applications. The three most influential transformer-based architectures are BERT, GPT, and T5, each representing different approaches to language modeling.
        </Text>
      </Stack>

      {/* BERT Architecture */}
      <Stack spacing="md">
        <Title order={2} id="bert-architecture">BERT: Bidirectional Encoder Representations from Transformers</Title>
        
        <Grid>
          <Grid.Col span={12}>
            <Stack spacing="md">
              <Text>
                BERT, introduced by Google in 2018, pioneered the use of bidirectional context in pre-trained language models. Unlike the original transformer, BERT uses only the encoder stack and is designed to develop rich bidirectional representations.
              </Text>
              
              <Box className="p-4 border rounded bg-gray-50">
                <Title order={4}>BERT Architecture Highlights</Title>
                <List>
                  <List.Item><strong>Architecture Type:</strong> Encoder-only transformer</List.Item>
                  <List.Item><strong>Bidirectionality:</strong> Attends to context from both left and right</List.Item>
                  <List.Item><strong>Pre-training Tasks:</strong> Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)</List.Item>
                  <List.Item><strong>Model Sizes:</strong> BERT-Base (110M parameters), BERT-Large (340M parameters)</List.Item>
                  <List.Item><strong>Special Tokens:</strong> BERT uses the WordPiece tokenization method</List.Item>
                </List>
              </Box>

              <Title order={3}>Pre-training Objectives</Title>
              <Text>
                BERT is pre-trained with two objectives:
              </Text>
              
              <List>
                <List.Item>
                  <strong>Masked Language Modeling (MLM):</strong> Randomly mask 15% of tokens in the input and train the model to predict these masked tokens. This forces the model to learn contextualized representations of words.
                  <BlockMath>{`
                    \\mathcal{L}_{MLM} = -\\mathbb{E}_{x \\sim X} \\log P(x_{masked} | x_{\\backslash masked})
                  `}</BlockMath>
                </List.Item>
                <List.Item>
                  <strong>Next Sentence Prediction (NSP):</strong> Given two sentences, predict whether the second sentence follows the first in the original text. This helps the model understand relationships between sentences.
                  <BlockMath>{`
                    \\mathcal{L}_{NSP} = -\\mathbb{E}_{(s_A, s_B) \\sim S} \\log P(\\text{IsNext}(s_A, s_B) | [s_A; s_B])
                  `}</BlockMath>
                </List.Item>
              </List>



              <Title order={3} mt="lg">Target Data Format and Applications</Title>
              <Text>
                BERT excels in tasks requiring rich understanding of context and relationships between elements in text.
              </Text>
              
              <Table withTableBorder withColumnBorders>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Task Type</Table.Th>
                    <Table.Th>Input Format</Table.Th>
                    <Table.Th>Examples</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  <Table.Tr>
                    <Table.Td>Sequence Classification</Table.Td>
                    <Table.Td><code>[CLS] text [SEP]</code></Table.Td>
                    <Table.Td>Sentiment analysis, topic classification</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Sentence Pair Tasks</Table.Td>
                    <Table.Td><code>[CLS] text A [SEP] text B [SEP]</code></Table.Td>
                    <Table.Td>Natural language inference, paraphrase detection</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Token Classification</Table.Td>
                    <Table.Td><code>[CLS] token₁ token₂ ... tokenₙ [SEP]</code></Table.Td>
                    <Table.Td>Named entity recognition, part-of-speech tagging</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Question Answering</Table.Td>
                    <Table.Td><code>[CLS] question [SEP] context [SEP]</code></Table.Td>
                    <Table.Td>SQuAD, extractive QA</Table.Td>
                  </Table.Tr>
                </Table.Tbody>
              </Table>
              
              <Text mt="md">
                BERT's bidirectional nature makes it particularly strong for tasks requiring deep language understanding, but less suited for open-ended text generation compared to autoregressive models.
              </Text>
            </Stack>
          </Grid.Col>
        </Grid>
      </Stack>

      {/* GPT Architecture */}
      <Stack spacing="md">
        <Title order={2} id="gpt-architecture">GPT: Generative Pre-trained Transformer</Title>
        
        <Grid>
          <Grid.Col span={12}>
            <Stack spacing="md">
              <Text>
                The Generative Pre-trained Transformer (GPT) family, developed by OpenAI, uses a decoder-only transformer architecture optimized for text generation. GPT models are trained to predict the next token in a sequence, enabling them to generate coherent and contextually relevant text.
              </Text>
              
              <Box className="p-4 border rounded bg-gray-50">
                <Title order={4}>GPT Architecture Highlights</Title>
                <List>
                  <List.Item><strong>Architecture Type:</strong> Decoder-only transformer</List.Item>
                  <List.Item><strong>Directionality:</strong> Unidirectional (left-to-right) attention</List.Item>
                  <List.Item><strong>Pre-training Task:</strong> Autoregressive Language Modeling</List.Item>
                  <List.Item><strong>Evolution:</strong> GPT-1 (117M), GPT-2 (1.5B), GPT-3 (175B), GPT-4 (estimated trillions)</List.Item>
                  <List.Item><strong>Context Window:</strong> Increasingly larger (GPT-3: 2048 tokens, GPT-4: 8K-32K tokens)</List.Item>
                </List>
              </Box>

              <Title order={3}>Pre-training Objective</Title>
              <Text>
                GPT models are trained with a causal language modeling objective, where they learn to predict the next token given all previous tokens:
              </Text>
              
              <BlockMath>{`
                \\mathcal{L}_{LM} = -\\sum_{i=1}^{n} \\log P(x_i | x_{<i}; \\theta)
              `}</BlockMath>
              
              <Text>
                Where <InlineMath>{`x_i`}</InlineMath> is the current token, <InlineMath>{`x_{<i}`}</InlineMath> represents all previous tokens, and <InlineMath>{`\\theta`}</InlineMath> are the model parameters. This training approach enforces the auto-regressive property where each token can only attend to previous tokens.
              </Text>

{/* For GPT section - Add after "Pre-training Objective" */}
<Title order={3} mt="lg">Tokenization Approach</Title>
<Text>
  The GPT family uses Byte-Pair Encoding (BPE) for tokenization, which iteratively merges the most frequent character or subword pairs to create a vocabulary of subword units. This allows the model to handle arbitrary text including rare words, code, and even non-English languages effectively.
</Text>

<Text>
  GPT-2 and GPT-3 both implement BPE with slight variations:
</Text>

<List>
  <List.Item><strong>GPT-2:</strong> Uses a 50,257 token vocabulary and preserves case, spacing, and punctuation</List.Item>
  <List.Item><strong>GPT-3:</strong> Uses a larger vocabulary of approximately 100K tokens, optimized for broad language coverage</List.Item>
  <List.Item><strong>GPT-4:</strong> Further improves tokenization efficiency with an enhanced BPE implementation</List.Item>
</List>

<Text>
  Unlike WordPiece used in BERT, GPT's BPE implementation doesn't use explicit markers for subword continuations, which affects how the model processes and generates text.
</Text>
              <Title order={3} mt="lg">Architectural Modifications in GPT Variants</Title>
              <Accordion variant="separated">
                <Accordion.Item value="gpt1">
                  <Accordion.Control>
                    <Text fw={500}>GPT-1: The Original</Text>
                  </Accordion.Control>
                  <Accordion.Panel>
                    <Text>
                      The first GPT model closely followed the transformer decoder structure with 12 layers and 117M parameters. It demonstrated the effectiveness of the pre-training + fine-tuning approach for NLP tasks.
                    </Text>
                  </Accordion.Panel>
                </Accordion.Item>
                
                <Accordion.Item value="gpt2">
                  <Accordion.Control>
                    <Text fw={500}>GPT-2: Scaling Up</Text>
                  </Accordion.Control>
                  <Accordion.Panel>
                    <Text>
                      GPT-2 increased the model size significantly (up to 1.5B parameters) and was trained on a more diverse dataset. It introduced layer normalization to the beginning of each sub-block and added an additional layer normalization after the final self-attention block.
                    </Text>
                  </Accordion.Panel>
                </Accordion.Item>
                
                <Accordion.Item value="gpt3">
                  <Accordion.Control>
                    <Text fw={500}>GPT-3: Massive Scale</Text>
                  </Accordion.Control>
                  <Accordion.Panel>
                    <Text>
                      GPT-3 scaled up to 175B parameters with 96 attention layers and implemented alternating dense and locally banded sparse attention patterns. It demonstrated emergent abilities including few-shot learning without parameter updates.
                    </Text>
                  </Accordion.Panel>
                </Accordion.Item>
                
                <Accordion.Item value="gpt4">
                  <Accordion.Control>
                    <Text fw={500}>GPT-4: Multimodal Capabilities</Text>
                  </Accordion.Control>
                  <Accordion.Panel>
                    <Text>
                      GPT-4 represents a further scaling in both parameters and training compute. It introduced multimodal capabilities (text and images) and improved alignment with human preferences through RLHF (Reinforcement Learning from Human Feedback).
                    </Text>
                  </Accordion.Panel>
                </Accordion.Item>
              </Accordion>

              <Title order={3} mt="lg">Target Data Format and Applications</Title>
              <Text>
                GPT models excel at generative tasks and can be prompted to perform a wide range of language functions.
              </Text>
              
              <Table withTableBorder withColumnBorders>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Task Type</Table.Th>
                    <Table.Th>Input/Output Pattern</Table.Th>
                    <Table.Th>Examples</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  <Table.Tr>
                    <Table.Td>Text Completion</Table.Td>
                    <Table.Td>Partial prompt → Completed text</Table.Td>
                    <Table.Td>Auto-completing sentences, paragraphs, or documents</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Creative Writing</Table.Td>
                    <Table.Td>Writing prompt → Story, poem, script</Table.Td>
                    <Table.Td>Fiction writing, poetry, screenplays</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Conversational AI</Table.Td>
                    <Table.Td>Dialog history → Next response</Table.Td>
                    <Table.Td>Chatbots, virtual assistants</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Few-shot Learning</Table.Td>
                    <Table.Td>Examples + new instance → Prediction</Table.Td>
                    <Table.Td>Classification, translation with minimal examples</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Code Generation</Table.Td>
                    <Table.Td>Code specification → Implementation</Table.Td>
                    <Table.Td>Programming assistance, code completion</Table.Td>
                  </Table.Tr>
                </Table.Tbody>
              </Table>
              
              <Text mt="md">
                GPT's autoregressive nature makes it particularly well-suited for text generation tasks, but the unidirectional attention limits its ability to understand bidirectional context compared to models like BERT.
              </Text>
            </Stack>
          </Grid.Col>
        </Grid>
      </Stack>

      {/* T5 Architecture */}
      <Stack spacing="md">
        <Title order={2} id="t5-architecture">T5: Text-to-Text Transfer Transformer</Title>
        
        <Grid>
          <Grid.Col span={12}>
            <Stack spacing="md">
              <Text>
                Text-to-Text Transfer Transformer (T5), developed by Google, unifies NLP tasks by framing them all as text-to-text problems. T5 uses the complete encoder-decoder architecture and introduces several innovations in pre-training and task formatting.
              </Text>
              
              <Box className="p-4 border rounded bg-gray-50">
                <Title order={4}>T5 Architecture Highlights</Title>
                <List>
                  <List.Item><strong>Architecture Type:</strong> Full encoder-decoder transformer</List.Item>
                  <List.Item><strong>Pre-training Task:</strong> Span corruption (similar to masked language modeling)</List.Item>
                  <List.Item><strong>Unified Framework:</strong> All NLP tasks formatted as text-to-text</List.Item>
                  <List.Item><strong>Model Sizes:</strong> Small (60M), Base (220M), Large (770M), XL (3B), XXL (11B)</List.Item>
                  <List.Item><strong>Variants:</strong> T5, mT5 (multilingual), ByT5 (byte-level), LongT5 (long documents)</List.Item>
                </List>
              </Box>

              <Title order={3}>Pre-training Objective</Title>
              <Text>
                T5 uses a "span corruption" pre-training objective, a variant of masked language modeling:
              </Text>
              
              <List ordered>
                <List.Item>Randomly select spans of text (average length 3 tokens)</List.Item>
                <List.Item>Replace each span with a single sentinel token (e.g., &lt;X&gt;)</List.Item>
                <List.Item>The model is trained to predict the original text of all corrupted spans, given the corrupted text</List.Item>
              </List>
              
              <CodeBlock language="text" code={`Input: "The quick brown fox jumps &lt;X&gt; the lazy dog."
Target: "&lt;X&gt; over"`}
              />
              
{/* For T5 section - Add after "Text-to-Text Framework" */}
<Title order={3} mt="lg">Tokenization Method</Title>
<Text>
  T5 uses SentencePiece tokenization, which treats text as a sequence of Unicode characters and applies subword segmentation without requiring pre-tokenization. This makes it language-agnostic and particularly effective for multilingual models.
</Text>

<Text>
  Key characteristics of T5's tokenization approach:
</Text>

<List>
  <List.Item><strong>Word boundary marking:</strong> Uses the "▁" prefix (underscore) to mark the beginning of words</List.Item>
  <List.Item><strong>Vocabulary size:</strong> Standard T5 uses a 32,000 token vocabulary, while mT5 (multilingual) uses 250,000 tokens</List.Item>
  <List.Item><strong>Special case - ByT5:</strong> Operates directly on bytes rather than subword tokens, using a vocabulary of just 256 tokens</List.Item>
</List>

<Text>
  This tokenization approach supports T5's span corruption objective and enables efficient processing of diverse languages and tasks.
</Text>
              <Title order={3} mt="lg">Text-to-Text Framework</Title>
              <Text>
                T5's key innovation is formatting all NLP tasks as text-to-text problems, making it possible to use the same model, loss function, and hyperparameters across a wide range of tasks.
              </Text>
              
              <BlockMath>{`
                \\text{output} = \\text{T5}(\\text{task prefix} + \\text{input})
              `}</BlockMath>
              
              <Title order={3} mt="lg">Task-Specific Formatting</Title>
              <Table withTableBorder withColumnBorders>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Task</Table.Th>
                    <Table.Th>Input Format</Table.Th>
                    <Table.Th>Output Format</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  <Table.Tr>
                    <Table.Td>Translation</Table.Td>
                    <Table.Td><code>translate English to German: {"{text}"}</code></Table.Td>
                    <Table.Td>Translated text in German</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Summarization</Table.Td>
                    <Table.Td><code>summarize: {"{document}"}</code></Table.Td>
                    <Table.Td>Summary text</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Question Answering</Table.Td>
                    <Table.Td><code>question: {"{question}"} context: {"{context}"}</code></Table.Td>
                    <Table.Td>Answer text</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Text Classification</Table.Td>
                    <Table.Td><code>classify sentiment: {"{text}"}</code></Table.Td>
                    <Table.Td>Class label (e.g., "positive")</Table.Td>
                  </Table.Tr>
                </Table.Tbody>
              </Table>
              <Title order={3} mt="lg">Architectural Innovations</Title>
              <List>
                <List.Item>
                  <strong>Simplified Relative Positional Encoding:</strong> T5 replaces the sinusoidal positional encoding with a simplified relative position embedding that only considers the relative distance between tokens up to a certain maximum distance.
                </List.Item>
                <List.Item>
                  <strong>Layer Normalization:</strong> T5 applies layer normalization before attention and feed-forward layers rather than after, which improves training stability.
                </List.Item>
                <List.Item>
                  <strong>GELU Activation:</strong> Uses Gaussian Error Linear Units instead of ReLU for better gradient flow.
                </List.Item>
              </List>

              <Title order={3} mt="lg">Target Data Format and Applications</Title>
              <Text>
                T5's encoder-decoder architecture makes it versatile for both understanding and generation tasks.
              </Text>
              
              <Box className="p-4 border rounded">
                <Title order={4}>T5 Strengths</Title>
                <Grid>
                  <Grid.Col span={6}>
                    <Title order={5}>Particularly Strong For:</Title>
                    <List>
                      <List.Item>Multi-task learning</List.Item>
                      <List.Item>Cross-lingual transfer</List.Item>
                      <List.Item>Summarization</List.Item>
                      <List.Item>Translation</List.Item>
                      <List.Item>Question answering</List.Item>
                    </List>
                  </Grid.Col>
                  <Grid.Col span={6}>
                    <Title order={5}>Key Advantages:</Title>
                    <List>
                      <List.Item>Consistent interface across tasks</List.Item>
                      <List.Item>Strong bidirectional understanding</List.Item>
                      <List.Item>Generative capabilities</List.Item>
                      <List.Item>Effective transfer learning</List.Item>
                    </List>
                  </Grid.Col>
                </Grid>
              </Box>
              
              <Text mt="md">
                T5's unified text-to-text approach allows it to handle a wide range of tasks with the same model architecture, making it particularly valuable for multi-task learning and transfer learning scenarios.
              </Text>
            </Stack>
          </Grid.Col>
        </Grid>
      </Stack>

      {/* Comparative Analysis */}
      <Stack spacing="md">
        <Title order={2} id="comparative-analysis">Comparative Analysis of Transformer Architectures</Title>
        
        <Text>
          Each transformer architecture makes different design choices that affect its capabilities and application domains.
        </Text>
        
        <Table withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Feature</Table.Th>
              <Table.Th>BERT</Table.Th>
              <Table.Th>GPT</Table.Th>
              <Table.Th>T5</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Architecture</Table.Td>
              <Table.Td>Encoder-only</Table.Td>
              <Table.Td>Decoder-only</Table.Td>
              <Table.Td>Encoder-Decoder</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Attention Mechanism</Table.Td>
              <Table.Td>Bidirectional</Table.Td>
              <Table.Td>Unidirectional (autoregressive)</Table.Td>
              <Table.Td>Bidirectional in encoder, Unidirectional in decoder</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Primary Pre-training Objective</Table.Td>
              <Table.Td>Masked Language Modeling</Table.Td>
              <Table.Td>Next Token Prediction</Table.Td>
              <Table.Td>Span Corruption</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Strengths</Table.Td>
              <Table.Td>Understanding, classification, token-level tasks</Table.Td>
              <Table.Td>Text generation, creative writing, conversational AI</Table.Td>
              <Table.Td>Multi-task learning, text-to-text transformations</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Limitations</Table.Td>
              <Table.Td>Limited generation capabilities</Table.Td>
              <Table.Td>Less context-aware than bidirectional models</Table.Td>
              <Table.Td>Higher computational requirements</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Ideal Input Data</Table.Td>
              <Table.Td>Classification tasks, NLU tasks</Table.Td>
              <Table.Td>Open-ended prompts, completion tasks</Table.Td>
              <Table.Td>Structured tasks with clear input/output formats</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
        
        <Title order={3} mt="lg">Architecture Diagrams</Title>
        
        <Grid>
          <Grid.Col span={4}>
            <Box className="p-4 border rounded text-center">
              <Title order={4}>BERT (Encoder-only)</Title>
              <pre className="mt-2">
{`    Input Tokens
         ↓
     Embeddings
         ↓
   [Self-Attention]
         ↓
  [Feed-Forward NN]
         ↓
   [Self-Attention]
         ↓
  [Feed-Forward NN]
         ↓
        ...
         ↓
   Final Embeddings
   [CLS]  T₁  T₂  ...`}
              </pre>
            </Box>
          </Grid.Col>
          
          <Grid.Col span={4}>
            <Box className="p-4 border rounded text-center">
              <Title order={4}>GPT (Decoder-only)</Title>
              <pre className="mt-2">
{`    Input Tokens
         ↓
     Embeddings
         ↓
 [Masked Self-Attention]
         ↓
  [Feed-Forward NN]
         ↓
 [Masked Self-Attention]
         ↓
  [Feed-Forward NN]
         ↓
        ...
         ↓
   Output Embeddings
         ↓
     Next Token`}
              </pre>
            </Box>
          </Grid.Col>
          
          <Grid.Col span={4}>
            <Box className="p-4 border rounded text-center">
              <Title order={4}>T5 (Encoder-Decoder)</Title>
              <pre className="mt-2">
{`    Input Tokens      Target Tokens
         ↓                  ↓
      Encoder            Decoder
  [Self-Attention]    [Masked Self-
         ↓                Attention]
  [Feed-Forward]            ↓
         ↓            [Cross-Attention]
        ...                 ↓
         ↓            [Feed-Forward]
   Encoder Output          ↓
         ↓                ...
         ↓                 ↓
         -----→-------  Output
                ↑
          Cross-Attention`}
              </pre>
            </Box>
          </Grid.Col>
        </Grid>
      </Stack>
    </Stack>
  );
};

export default TransformerArchitectures;