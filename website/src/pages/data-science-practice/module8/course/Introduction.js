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
            src="/assets/data-science-practice/module8/Hierarchical-nlp.png"
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
            src="/assets/data-science-practice/module8/NatureofText.png"
            alt="Visualization of sequential dependencies in text"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
        </Flex>

      </div>

      <div data-slide>
        <Title order={2}>Character Encodings and Text Representation</Title>

        <Text mt="md">
          Text processing begins with understanding how characters are encoded and represented digitally.
        </Text>

        <Title order={3} mt="lg">ASCII and Extended Character Sets</Title>
        <Text mt="sm">
          ASCII (American Standard Code for Information Interchange) represents 128 characters using 7 bits:
          uppercase letters (A-Z: 65-90), lowercase letters (a-z: 97-122), digits (0-9: 48-57), and special characters.
        </Text>

        <Text mt="md">
          The case difference is systematic: uppercase 'A' (65) and lowercase 'a' (97) differ by 32.
          This property enables efficient case transformations but introduces complexity for case-insensitive matching.
        </Text>
                <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/ascii.webp"
            alt="ASCII"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
        </Flex>

        <Title order={3} mt="lg">Unicode and Multilingual Text</Title>
        <Text mt="sm">
          Unicode extends beyond ASCII to support diverse writing systems: Latin scripts (English, French, German),
          non-Latin alphabets (Cyrillic, Greek, Arabic, Hebrew), logographic systems (Chinese, Japanese), and complex scripts
          (Devanagari, Thai). UTF-8 encoding uses 1-4 bytes per character, making it variable-length and space-efficient.
        </Text>

        <Text mt="md">
          Special characters present unique challenges: punctuation varies by language (English period vs. Arabic question mark),
          whitespace includes spaces, tabs, and line breaks, diacritics modify base characters (é, ñ, ü),
          and emojis require multi-byte encoding (🙂 = U+1F642).
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/utf.png"
            alt="Comparison of ASCII and Unicode character encodings with examples"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Language Diversity at Scale</Title>

        <Text mt="md">
          Human language exhibits remarkable diversity across multiple dimensions, creating substantial challenges for NLP systems.
        </Text>

        <Title order={3} mt="lg">Languages and Writing Systems</Title>
        <List spacing="sm" mt="sm">
          <List.Item><strong>~7,000 living languages</strong> worldwide (Ethnologue 2023)</List.Item>
          <List.Item><strong>~300 writing systems</strong> across history, with ~150 currently in use</List.Item>
          <List.Item><strong>26 letters</strong> in English alphabet vs. <strong>50,000+ characters</strong> in Chinese writing system</List.Item>
        </List>

        <Title order={3} mt="lg">Vocabulary and Text Volume</Title>
        <List spacing="sm" mt="sm">
          <List.Item><strong>170,000+ words</strong> in current English usage (Oxford English Dictionary)</List.Item>
          <List.Item><strong>~130 million books</strong> published in all languages throughout history</List.Item>
          <List.Item><strong>2.5 million+ books</strong> published annually worldwide</List.Item>
          <List.Item><strong>Billions of web pages</strong> containing trillions of words across languages</List.Item>
        </List>


        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/world-of-languages-large.png"
            alt="Visualization of language diversity statistics and writing system examples"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Global language diversity: distribution of speakers, writing systems, and text resources
          </Text>
        </Flex>

        <Text mt="lg" size="sm" fs="italic">
          References: Ethnologue (2023) - https://www.ethnologue.com; Unicode Standard v15.1 (2023) - https://unicode.org;
          Oxford English Dictionary - https://oed.com; Google Books Library Project (2010) - http://booksearch.blogspot.com/2010/08/books-of-world.html
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Sequence Representations Beyond Natural Language</Title>

        <Text mt="md">
          The principles of text processing extend to other sequential symbolic data with finite alphabets.
        </Text>

        <Title order={3} mt="lg">Biological Sequences</Title>
        <Text mt="sm">
          DNA sequences use four nucleotide bases: A (Adenine), T (Thymine), G (Guanine), C (Cytosine).
          These sequences encode genetic information where order determines biological function.
          Protein sequences use 20 amino acids represented by single-letter codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).
        </Text>

        <Title order={3} mt="lg">Musical Sequences</Title>
        <Text mt="sm">
          Music can be represented symbolically through MIDI note numbers (0-127), pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B),
          duration values (whole, half, quarter notes), and ABC notation for folk music transcription.
          Sequential models can learn patterns in melodies, harmonies, and compositional structures.
        </Text>

        <Title order={3} mt="lg">Chemical Representations</Title>
        <Text mt="sm">
          SMILES (Simplified Molecular Input Line Entry System) notation represents molecular structures as text strings
          using characters for atoms (C, N, O, S), bonds (-, =, #), and branches (parentheses).
          InChI provides standardized chemical structure representations for database searching and compound identification.
        </Text>

        <Title order={3} mt="lg">Mathematical Expressions</Title>
        <Text mt="sm">
          Mathematical notation forms a symbolic language with operators (+, -, ×, ÷, =), variables (x, y, z),
          functions (sin, cos, log), Greek letters (α, β, γ), and special symbols (∫, ∑, ∂).
          LaTeX provides a standardized text representation for mathematical expressions, enabling symbolic computation and automated theorem proving.
        </Text>

        <Title order={3} mt="lg">Programming Code</Title>
        <Text mt="sm">
          Source code combines natural language elements (variable names, comments) with formal syntax rules (keywords, operators, delimiters).
          Programming languages like Python, Java, and C++ have finite token vocabularies including keywords (if, while, return),
          operators (=, +, ==), and punctuation (braces, parentheses, semicolons). Code exhibits sequential dependencies through control flow,
          function calls, and variable scope.
        </Text>

        <Text mt="lg">
          These diverse sequence types share fundamental properties with natural language: discrete vocabularies,
          sequential dependencies, and meaningful patterns that can be learned through similar computational approaches.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/dna.jpg"
            alt="Examples of non-language sequences: DNA"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
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


        <Text mt="lg" size="sm" fs="italic">
          Reference: Vaswani et al., "Attention Is All You Need" (2017) - https://arxiv.org/abs/1706.03762
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>NLP Task Categories</Title>
                <Text mt="lg">
          Text sequences can be represented as:
        </Text>

        <BlockMath>
          {`\\mathcal{S} = (w_1, w_2, \\ldots, w_n) \\quad \\text{where } w_i \\in \\mathcal{V}`}
        </BlockMath>

        <Text mt="md">
          Where <InlineMath math="\mathcal{S}" /> is a sequence and <InlineMath math="\mathcal{V}" /> is the vocabulary.
        </Text>

        <Title order={3} mt="md">Text to Value Tasks</Title>

        <Text mt="sm"><strong>Text Classification</strong></Text>
        <BlockMath>{`f: \\mathcal{S} \\rightarrow \\{1,\\ldots,K\\}`}</BlockMath>
        <Text size="sm">Maps text sequences to K discrete categories (sentiment analysis, topic classification, spam detection)</Text>

        <Text mt="md"><strong>Regression from Text</strong></Text>
        <BlockMath>{`f: \\mathcal{S} \\rightarrow \\mathbb{R}^n`}</BlockMath>
        <Text size="sm">Predicts continuous values from text (price prediction, readability scores)</Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/sentimentanalysis.avif"
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
        <Title order={2}>Text to Text Tasks</Title>

        <Text mt="md"><strong>Machine Translation</strong></Text>
        <BlockMath>{`f: \\mathcal{S}_{L1} \\rightarrow \\mathcal{S}_{L2}`}</BlockMath>
        <Text size="sm">Converts text from source language L1 to target language L2</Text>

        <Text mt="lg"><strong>Text Summarization</strong></Text>
        <BlockMath>{`f: \\mathcal{S}_{long} \\rightarrow \\mathcal{S}_{short}`}</BlockMath>
        <Text size="sm">Generates concise representation of longer text</Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/text2text.png"
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

        <Text mt="md"><strong>Language Modeling</strong></Text>
        <Text size="sm" mt="xs">Models probability distribution over sequences through self-supervised learning</Text>

        <Text mt="lg" size="sm"><em>Causal Language Modeling (CLM):</em></Text>
        <BlockMath>{`P(w_t | w_1, w_2, \\ldots, w_{t-1})`}</BlockMath>
        <Text size="sm">Predicts next token given previous tokens (autoregressive, left-to-right)</Text>
        <Text size="sm">Used in GPT models for text generation</Text>

        <Text mt="md" size="sm"><em>Masked Language Modeling (MLM):</em></Text>
        <BlockMath>{`P(w_i | w_1, \\ldots, w_{i-1}, w_{i+1}, \\ldots, w_n)`}</BlockMath>
        <Text size="sm">Predicts masked tokens using bidirectional context</Text>
        <Text size="sm">Used in BERT models for text understanding</Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/dentistygpt.png"
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
  
  <Text size="sm" mt="xs"><em>Optical Character Recognition (OCR):</em></Text>
  <BlockMath>{`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\mathcal{S}`}</BlockMath>
  <Text size="sm">Extract text from images or documents</Text>

  <Text size="sm" mt="sm"><em>Visual Question Answering:</em></Text>
  <BlockMath>{`f: (\\mathbb{R}^{H \\times W \\times C}, \\mathcal{S}_{\\text{question}}) \\rightarrow \\mathcal{S}_{\\text{answer}}`}</BlockMath>
  <Text size="sm">Answer questions about image content</Text>

  <Text size="sm" mt="sm"><em>Text-to-Image Generation:</em></Text>
  <BlockMath>{`f: \\mathcal{S} \\rightarrow \\mathbb{R}^{H \\times W \\times C}`}</BlockMath>
  <Text size="sm">Generate image from text description</Text>

  <Text mt="lg"><strong>Speech-Text Processing</strong></Text>
  
  <Text size="sm" mt="xs"><em>Speech Recognition (ASR):</em></Text>
  <BlockMath>{`f: \\mathcal{A} \\rightarrow \\mathcal{S}`}</BlockMath>
  <Text size="sm">Transcribe audio to text</Text>

  <Text size="sm" mt="sm"><em>Text-to-Speech (TTS):</em></Text>
  <BlockMath>{`f: \\mathcal{S} \\rightarrow \\mathcal{A}`}</BlockMath>
  <Text size="sm">Synthesize speech from text</Text>

  <Flex direction="column" align="center" mt="xl" mb="md">
    <Image
      src="/assets/data-science-practice/module8/multimodal.jpg"
      alt="Examples of multimodal NLP tasks combining vision, language, and audio"
      style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
      fluid
      mb="sm"
    />
  </Flex>
</div>
    </>
  );
};

export default Introduction;



