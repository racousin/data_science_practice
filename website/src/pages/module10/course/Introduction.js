import React from "react";
import { Row, Col, Card, Table } from "react-bootstrap";
import { Text, Title, Group, Image, Stack, Container } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import DataInteractionPanel from "components/DataInteractionPanel";

// TaskBox component for displaying NLP tasks with mathematical formulation
const TaskBox = ({ title, formula, description }) => (
  <Card className="h-100">
    <Card.Body>
      <Title order={4}>{title}</Title>
      <div className="my-2">
        <BlockMath math={formula} />
      </div>
      <Text size="sm">{description}</Text>
    </Card.Body>
  </Card>
);

const Introduction = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md" id="introduction">Introduction to Natural Language Processing</Title>
      
      <Stack spacing="xl">
        <Text mb="lg">
          Natural Language Processing (NLP) sits at the intersection of linguistics, computer science, and artificial intelligence, 
          focused on enabling computers to understand, interpret, and generate human language. This field has evolved dramatically 
          from rule-based systems to sophisticated deep learning models that can translate languages, answer questions, 
          summarize documents, and generate human-like text.
        </Text>
        
        {/* Text Data Representation */}
        <div>
          <Title order={2} id="text-representation">Text Data Representation</Title>
          
          
          <Text className="mt-3">
            At its core, text is sequential, discrete, and unstructured data that requires transformation into numerical 
            representations before it can be processed by machine learning algorithms.
          </Text>

          {/* Levels of Text Structure */}
          <Title order={3} className="mt-3">Levels of Text Structure</Title>
          <Card className="mb-3">
            <Card.Body>
              <Row>
                <Col md={6}>
                  <ul>
                    <li><strong>Characters:</strong> Individual letters, numbers, punctuation (e.g., 'a', '5', '!')</li>
                    <li><strong>Subwords:</strong> Character sequences that form meaningful units (e.g., 'ing', 'pre-', 'un-')</li>
                    <li><strong>Words:</strong> Complete lexical units (e.g., 'language', 'processing')</li>
                  </ul>
                </Col>
                <Col md={6}>
                  <ul>
                    <li><strong>Sentences:</strong> Sequences of words with complete meaning</li>
                    <li><strong>Paragraphs:</strong> Related sentences grouped together</li>
                    <li><strong>Documents:</strong> Complete texts with overall structure and context</li>
                  </ul>
                </Col>
              </Row>
            </Card.Body>
          </Card>

          {/* Text Structure Types */}
          <Title order={3}>Text Structure Types</Title>
          <Group spacing="xs" className="bg-gray-100 p-4 rounded-lg mb-3">
            <Stack spacing={0} style={{ width: '100%' }}>
              <Row>
                <Col md={4}>
                  <Title order={5}>Structured Text</Title>
                  <Text size="sm">
                    Follows consistent format and organization:
                    • Database records
                    • XML/JSON documents
                    • Forms and templates
                    • Tables and CSV files
                  </Text>
                </Col>
                <Col md={4}>
                  <Title order={5}>Semi-structured Text</Title>
                  <Text size="sm">
                    Contains some organizational elements:
                    • Email (headers + body)
                    • Social media posts (metadata + content)
                    • HTML web pages
                    • Academic papers with sections
                  </Text>
                </Col>
                <Col md={4}>
                  <Title order={5}>Unstructured Text</Title>
                  <Text size="sm">
                    Free-form with minimal explicit organization:
                    • Conversational text
                    • Novels and stories
                    • Customer reviews
                    • Transcribed speech
                  </Text>
                </Col>
              </Row>
            </Stack>
          </Group>

          {/* Sequential Nature */}
          <Title order={3}>Sequential Nature of Text</Title>
          <Card className="mb-3">
            <Card.Body>
              <Text>
                Text is inherently sequential, with meaning derived from the order of elements:
              </Text>
              <ul>
                <li>Word order impacts meaning: "Dog bites man" ≠ "Man bites dog"</li>
                <li>Dependencies can span across long distances in a sequence</li>
                <li>Context is critical for disambiguation (e.g., "bank" can mean financial institution or river edge)</li>
              </ul>
              <Text>
                Mathematically, text sequences can be represented as:
              </Text>
              <BlockMath>
                {`\\mathcal{S} = (w_1, w_2, \\ldots, w_n) \\quad \\text{where } w_i \\in \\mathcal{V}`}
              </BlockMath>
              <Text>
                Where <InlineMath math="\mathcal{S}" /> is a sequence and <InlineMath math="\mathcal{V}" /> is the vocabulary.
              </Text>
            </Card.Body>
          </Card>

        </div>
      </Stack>

      <Title order={2} mb="sm" id="history">Historical Context of NLP</Title>
      <Card className="mb-4">
        <Card.Body>
          <Text>
            NLP development has progressed through several distinct paradigms over the decades:
          </Text>
          <Table striped bordered hover className="mt-3">
            <thead>
              <tr>
                <th>Era</th>
                <th>Period</th>
                <th>Key Characteristics</th>
                <th>Notable Systems/Approaches</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Rule-based Era</td>
                <td>1950s-1980s</td>
                <td>Hard-coded linguistic rules and pattern matching</td>
                <td>ELIZA, SHRDLU, expert systems</td>
              </tr>
              <tr>
                <td>Statistical Era</td>
                <td>1990s-2000s</td>
                <td>Statistical models and machine learning</td>
                <td>N-gram models, Hidden Markov Models, IBM Translation Models</td>
              </tr>
              <tr>
                <td>Neural Network Era</td>
                <td>2010s-2018</td>
                <td>Word embeddings and recurrent neural architectures</td>
                <td>Word2Vec, GloVe, LSTM, GRU, Seq2Seq models</td>
              </tr>
              <tr>
                <td>Transformer Era</td>
                <td>2018-Present</td>
                <td>Self-attention and large pre-trained models</td>
                <td>BERT, GPT, T5, Transformer-based architectures</td>
              </tr>
            </tbody>
          </Table>
        </Card.Body>
      </Card>

      <Text mb="lg">
        The 2017 introduction of the Transformer architecture in "Attention Is All You Need" by Vaswani et al. marked a 
        pivotal moment in NLP, enabling models to process text in parallel rather than sequentially, leading to unprecedented 
        capabilities and scale.
      </Text>

      <Title order={2} mb="sm" id="applications">NLP Applications and Tasks</Title>
      
      <Stack spacing="xl" className="mt-4">
        <Text>
          Natural language data can serve as both input and output across diverse machine learning tasks,
          each with distinct formulations and applications.
        </Text>

        {/* Text to Value Tasks */}
        <div>
          <Title order={3} className="mb-3">Text to Value Tasks</Title>
          <Group grow>
            <TaskBox
              title="Text Classification"
              formula={`f: \\mathcal{S} \\rightarrow \\{1,\\ldots,K\\}`}
              description="Maps text sequences to K discrete categories. Examples: sentiment analysis, topic classification, spam detection, intent recognition."
            />
            <TaskBox
              title="Regression from Text"
              formula={`f: \\mathcal{S} \\rightarrow \\mathbb{R}^n`}
              description="Predicts continuous values from text. Examples: price prediction, readability scores, emotion intensity estimation."
            />
          </Group>
        </div>

        {/* Sequence Labeling Tasks */}
        <div>
          <Title order={3} className="mb-3">Sequence Labeling Tasks</Title>
          <Group grow>
            <TaskBox
              title="Token Classification"
              formula={`f: \\{w_1,\\ldots,w_n\\} \\rightarrow \\{c_1,\\ldots,c_n\\}`}
              description="Assigns a label to each token in a sequence. Examples: Named Entity Recognition (NER), Part-of-Speech (POS) tagging."
            />
            <TaskBox
              title="Span Detection"
              formula={`f: \\mathcal{S} \\rightarrow \\{(s_i, e_i, c_i)\\}_{i=1}^N`}
              description="Identifies spans of text with start positions (s), end positions (e), and classes (c). Examples: chunking, aspect extraction, mention detection."
            />
          </Group>
        </div>

        {/* Text to Text Tasks */}
        <div>
          <Title order={3} className="mb-3">Text to Text Tasks</Title>
          <Group grow>
            <TaskBox
              title="Machine Translation"
              formula={`f: \\mathcal{S}_{L1} \\rightarrow \\mathcal{S}_{L2}`}
              description="Converts text from source language L1 to target language L2. Examples: English-to-French translation, code translation."
            />
            <TaskBox
              title="Text Summarization"
              formula={`f: \\mathcal{S}_{long} \\rightarrow \\mathcal{S}_{short}`}
              description="Generates concise representation of longer text. Subtypes: extractive summarization (selects key sentences) and abstractive summarization (generates novel text)."
            />
          </Group>
        </div>

        {/* Generative Tasks */}
        <div>
          <Title order={3} className="mb-3">Generative Language Tasks</Title>
          <Group grow>
            <TaskBox
              title="Conditional Text Generation"
              formula={`f: \\mathcal{C} \\rightarrow \\mathcal{S}`}
              description="Generates text based on a conditioning input. Examples: text completion, dialogue systems, code generation, style transfer, prompt-based generation."
            />
            <TaskBox
              title="Language Modeling"
              formula={`P(w_t | w_1, w_2, \\ldots, w_{t-1})`}
              description="Models probability distribution over sequences by predicting next token given previous tokens. Foundation for many generative tasks."
            />
          </Group>
        </div>

        {/* Multimodal Tasks */}
        <div>
          <Title order={3} className="mb-3">Multimodal NLP Tasks</Title>
          <Group grow>
            <TaskBox
              title="Vision-Language Tasks"
              formula={`f: (\\mathbb{R}^{H \\times W \\times C}, \\mathcal{S}) \\rightarrow \\mathcal{Y}`}
              description="Combines image and text processing. Examples: image captioning, visual question answering, visual reasoning, text-to-image generation."
            />
            <TaskBox
              title="Speech-Text Processing"
              formula={`f: (\\mathcal{A}, \\mathcal{S}) \\rightarrow \\mathcal{Y}`}
              description="Processes both audio and text. Examples: speech recognition, text-to-speech, speech translation, voice assistants."
            />
          </Group>
        </div>
      </Stack>
    </Container>
  );
};

export default Introduction;