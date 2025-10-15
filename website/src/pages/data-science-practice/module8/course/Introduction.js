import React from "react";
import { Text, Title, List, Table, Flex, Image } from '@mantine/core';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const Introduction = () => {
  return (
    <>
      <div data-slide>
        <Title order={1}>Natural Language Processing</Title>

        <Text mt="md">
          Natural Language Processing (NLP) sits at the intersection of linguistics, computer science, and artificial intelligence,
          focused on enabling computers to understand, interpret, and generate human language.
        </Text>

        <Text mt="md">
          This field has evolved dramatically from rule-based systems to sophisticated deep learning models that can translate languages,
          answer questions, summarize documents, and generate human-like text.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Text as Sequential Data</Title>

        <Text mt="md">
          At its core, text is sequential, discrete, and unstructured data that requires transformation into numerical
          representations before processing by machine learning algorithms.
        </Text>

        <Title order={3} mt="lg">Levels of Text Structure</Title>

        <List spacing="sm" mt="md">
          <List.Item><strong>Characters:</strong> Individual letters, numbers, punctuation (e.g., 'a', '5', '!')</List.Item>
          <List.Item><strong>Subwords:</strong> Character sequences that form meaningful units (e.g., 'ing', 'pre-', 'un-')</List.Item>
          <List.Item><strong>Words:</strong> Complete lexical units (e.g., 'language', 'processing')</List.Item>
          <List.Item><strong>Sentences:</strong> Sequences of words with complete meaning</List.Item>
          <List.Item><strong>Paragraphs:</strong> Related sentences grouped together</List.Item>
          <List.Item><strong>Documents:</strong> Complete texts with overall structure and context</List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/text-structure-hierarchy.png"
            alt="Hierarchical levels of text structure from characters to documents"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Hierarchical organization of text from characters to complete documents
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Sequential Nature of Text</Title>

        <Text mt="md">
          Text is inherently sequential, with meaning derived from the order of elements:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>Word order impacts meaning: "Dog bites man" ≠ "Man bites dog"</List.Item>
          <List.Item>Dependencies can span across long distances in a sequence</List.Item>
          <List.Item>Context is critical for disambiguation (e.g., "bank" can mean financial institution or river edge)</List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/sequential-dependencies.png"
            alt="Visualization of sequential dependencies in text"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Long-range dependencies and word order effects in text sequences
          </Text>
        </Flex>

        <Text mt="lg">
          Mathematically, text sequences can be represented as:
        </Text>

        <BlockMath>
          {`\\mathcal{S} = (w_1, w_2, \\ldots, w_n) \\quad \\text{where } w_i \\in \\mathcal{V}`}
        </BlockMath>

        <Text mt="md">
          Where <InlineMath math="\mathcal{S}" /> is a sequence and <InlineMath math="\mathcal{V}" /> is the vocabulary.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Text Structure Types</Title>

        <Title order={3} mt="md">Structured Text</Title>
        <Text>
          Follows consistent format and organization: database records, XML/JSON documents, forms and templates, tables and CSV files.
        </Text>

        <Title order={3} mt="lg">Semi-structured Text</Title>
        <Text>
          Contains some organizational elements: email (headers + body), social media posts (metadata + content), HTML web pages,
          academic papers with sections.
        </Text>

        <Title order={3} mt="lg">Unstructured Text</Title>
        <Text>
          Free-form with minimal explicit organization: conversational text, novels and stories, customer reviews, transcribed speech.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/text-structure-types.png"
            alt="Examples of structured, semi-structured, and unstructured text"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Comparison of structured, semi-structured, and unstructured text formats
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Historical Evolution of NLP</Title>

        <Text mt="md">
          NLP development has progressed through several distinct paradigms:
        </Text>

        <Table striped mt="lg">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Era</Table.Th>
              <Table.Th>Period</Table.Th>
              <Table.Th>Key Characteristics</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Rule-based</Table.Td>
              <Table.Td>1950s-1980s</Table.Td>
              <Table.Td>Hard-coded linguistic rules and pattern matching</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Statistical</Table.Td>
              <Table.Td>1990s-2000s</Table.Td>
              <Table.Td>Statistical models and machine learning</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Neural Network</Table.Td>
              <Table.Td>2010s-2018</Table.Td>
              <Table.Td>Word embeddings and recurrent architectures</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Transformer</Table.Td>
              <Table.Td>2018-Present</Table.Td>
              <Table.Td>Self-attention and large pre-trained models</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/nlp-evolution-timeline.png"
            alt="Timeline of NLP evolution from rule-based to transformer era"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Evolution of NLP approaches from 1950s to present
          </Text>
        </Flex>

        <Text mt="lg" size="sm" fs="italic">
          Reference: Vaswani et al., "Attention Is All You Need" (2017) - https://arxiv.org/abs/1706.03762
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>NLP Task Categories</Title>

        <Title order={3} mt="md">Text to Value Tasks</Title>

        <Text mt="sm"><strong>Text Classification</strong></Text>
        <BlockMath>{`f: \\mathcal{S} \\rightarrow \\{1,\\ldots,K\\}`}</BlockMath>
        <Text size="sm">Maps text sequences to K discrete categories (sentiment analysis, topic classification, spam detection)</Text>

        <Text mt="md"><strong>Regression from Text</strong></Text>
        <BlockMath>{`f: \\mathcal{S} \\rightarrow \\mathbb{R}^n`}</BlockMath>
        <Text size="sm">Predicts continuous values from text (price prediction, readability scores)</Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/text-to-value-tasks.png"
            alt="Examples of text classification and regression tasks"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Text-to-value mappings: classification and regression examples
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Sequence Labeling Tasks</Title>

        <Text mt="md"><strong>Token Classification</strong></Text>
        <BlockMath>{`f: \\{w_1,\\ldots,w_n\\} \\rightarrow \\{c_1,\\ldots,c_n\\}`}</BlockMath>
        <Text size="sm">Assigns a label to each token (Named Entity Recognition, Part-of-Speech tagging)</Text>

        <Text mt="lg"><strong>Span Detection</strong></Text>
        <BlockMath>{`f: \\mathcal{S} \\rightarrow \\{(s_i, e_i, c_i)\\}_{i=1}^N`}</BlockMath>
        <Text size="sm">Identifies spans with start positions (s), end positions (e), and classes (c)</Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/sequence-labeling-examples.png"
            alt="Examples of token classification and span detection with NER and POS tagging"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Token-level labeling: NER and POS tagging examples
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Text to Text Tasks</Title>

        <Text mt="md"><strong>Machine Translation</strong></Text>
        <BlockMath>{`f: \\mathcal{S}_{L1} \\rightarrow \\mathcal{S}_{L2}`}</BlockMath>
        <Text size="sm">Converts text from source language L1 to target language L2</Text>

        <Text mt="lg"><strong>Text Summarization</strong></Text>
        <BlockMath>{`f: \\mathcal{S}_{long} \\rightarrow \\mathcal{S}_{short}`}</BlockMath>
        <Text size="sm">Generates concise representation of longer text</Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/text-to-text-tasks.png"
            alt="Examples of machine translation and summarization"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Sequence-to-sequence transformations: translation and summarization
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Generative Tasks</Title>

        <Text mt="md"><strong>Conditional Text Generation</strong></Text>
        <BlockMath>{`f: \\mathcal{C} \\rightarrow \\mathcal{S}`}</BlockMath>
        <Text size="sm">Generates text based on conditioning input (dialogue systems, code generation, style transfer)</Text>

        <Text mt="lg"><strong>Language Modeling</strong></Text>
        <BlockMath>{`P(w_t | w_1, w_2, \\ldots, w_{t-1})`}</BlockMath>
        <Text size="sm">Models probability distribution over sequences by predicting next token</Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/generative-tasks.png"
            alt="Examples of conditional text generation and language modeling"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Generative modeling: conditional generation and autoregressive prediction
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Multimodal NLP Tasks</Title>

        <Text mt="md"><strong>Vision-Language Tasks</strong></Text>
        <BlockMath>{`f: (\\mathbb{R}^{H \\times W \\times C}, \\mathcal{S}) \\rightarrow \\mathcal{Y}`}</BlockMath>
        <Text size="sm">Combines image and text processing (image captioning, visual question answering, text-to-image generation)</Text>

        <Text mt="lg"><strong>Speech-Text Processing</strong></Text>
        <BlockMath>{`f: (\\mathcal{A}, \\mathcal{S}) \\rightarrow \\mathcal{Y}`}</BlockMath>
        <Text size="sm">Processes both audio and text (speech recognition, text-to-speech, voice assistants)</Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/multimodal-nlp-tasks.png"
            alt="Examples of multimodal NLP tasks combining vision, language, and audio"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Multimodal integration: vision-language and speech-text processing
          </Text>
        </Flex>
      </div>
    </>
  );
};

export default Introduction;
