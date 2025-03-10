import React from "react";
import { 
  Container, 
  Title, 
  Text, 
  Table, 
  Accordion, 
  Card, 
  Grid, 
  Box, 
  List 
} from "@mantine/core";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from 'components/CodeBlock';


const SectionDivider = () => {return }

const Preprocessing = () => {
  const learningObjectives = [
    "Understand and apply common text cleaning techniques",
    "Compare different tokenization approaches and their applications",
    "Implement tokenization using modern NLP libraries",
    "Select appropriate tokenization strategies for specific NLP tasks",
    "Handle sequence length and special tokens effectively"
  ];

  return (
    <Container fluid>
      <Title order={1} mb="md">Text Preprocessing and Tokenization</Title>
      

      <Text mb="lg">
        Text preprocessing is a crucial first step in any NLP pipeline that transforms raw text 
        into a format suitable for model consumption. The quality of preprocessing significantly 
        impacts downstream model performance. This module covers essential text cleaning techniques 
        and tokenization approaches used in modern NLP systems.
      </Text>
      
      {/* Text Cleaning Section */}
      <SectionDivider />
      <Title order={2} id="cleaning" mb="sm">Text Cleaning Techniques</Title>
      
      <Text mb="md">
        Raw text data often contains noise, inconsistencies, and irrelevant information that can 
        negatively impact model performance. The following techniques help clean and normalize text data:
      </Text>
      
      <Card shadow="sm" p="md" mb="lg">
        <Table>
          <thead>
            <tr>
              <th>Technique</th>
              <th>Description</th>
              <th>Example</th>
              <th>When to Use</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Lowercasing</td>
              <td>Converting all text to lowercase</td>
              <td>"Hello World" → "hello world"</td>
              <td>Most cases, unless case carries meaning</td>
            </tr>
            <tr>
              <td>Punctuation removal</td>
              <td>Removing punctuation marks</td>
              <td>"Hello, world!" → "Hello world"</td>
              <td>When punctuation isn't semantically important</td>
            </tr>
            <tr>
              <td>Number normalization</td>
              <td>Replacing numbers with placeholders</td>
              <td>"I have 5 apples" → "I have [NUM] apples"</td>
              <td>When exact numbers aren't important</td>
            </tr>
            <tr>
              <td>Stopword removal</td>
              <td>Removing common words like "the", "is"</td>
              <td>"This is a test" → "This test"</td>
              <td>For specific applications like keyword extraction</td>
            </tr>
            <tr>
              <td>Stemming</td>
              <td>Reducing words to their root form</td>
              <td>"running" → "run"</td>
              <td>Simple applications with limited vocabulary</td>
            </tr>
            <tr>
              <td>Lemmatization</td>
              <td>Reducing words to dictionary form</td>
              <td>"better" → "good"</td>
              <td>When linguistic accuracy is important</td>
            </tr>
            <tr>
              <td>URL/email replacement</td>
              <td>Replacing URLs and emails with tokens</td>
              <td>"Visit https://example.com" → "Visit [URL]"</td>
              <td>When exact URLs aren't relevant</td>
            </tr>
            <tr>
              <td>Whitespace normalization</td>
              <td>Standardizing whitespace</td>
              <td>"Hello    world" → "Hello world"</td>
              <td>Always</td>
            </tr>
          </tbody>
        </Table>
      </Card>
      
      <Text mb="md">
        The choice of cleaning techniques depends on your specific task and domain. Modern transformer-based
        models are more robust to noisy text and often require minimal preprocessing compared to 
        traditional approaches.
      </Text>
      
      <CodeBlock
        language="python"
        code={`
# Basic text cleaning function
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text, lowercase=True, remove_punct=True, 
               remove_stopwords=False, lemmatize=False):
    """
    Cleans text by applying various preprocessing techniques.
    
    Args:
        text (str): Input text to clean
        lowercase (bool): Whether to convert to lowercase
        remove_punct (bool): Whether to remove punctuation
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to lemmatize words
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation
    if remove_punct:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into text
    cleaned_text = ' '.join(words)
    
    return cleaned_text

# Example usage
text = "Hello, world! This is an example of text cleaning."
cleaned = clean_text(text, lowercase=True, remove_punct=True, 
                     remove_stopwords=True, lemmatize=True)
print(cleaned)  # Output: hello world example text cleaning
`}
      />

      <Text mt="md" mb="md">
        For modern NLP pipelines with transformer-based models, minimal preprocessing is often 
        sufficient as these models are pre-trained on raw text and can handle linguistic variations.
      </Text>
      
      <CodeBlock
        language="python"
        code={`
# Minimal preprocessing for transformer models
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_for_transformer(text):
    """
    Minimal preprocessing for transformer models.
    Just handles whitespace normalization and control characters.
    """
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Remove control characters
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch == '\n')
    
    return text

text = "Hello,    world! \tThis is an example."
preprocessed = preprocess_for_transformer(text)
print(preprocessed)  # Output: Hello, world! This is an example.

# The tokenizer handles the rest
tokens = tokenizer(preprocessed)
print(tokens)
`}
      />
      
      {/* Tokenization Approaches Section */}
      <SectionDivider />
      <Title order={2} id="tokenization" mb="sm">Tokenization Approaches</Title>
      
      <Text mb="md">
        Tokenization is the process of converting text into tokens (discrete units) that can be 
        processed by NLP models. The choice of tokenization approach impacts the model's ability 
        to capture linguistic patterns.
      </Text>
      
      <Accordion mb="lg">
        <Accordion.Item value="word-tokenization">
          <Accordion.Control>
            <Text fw={600}>Word-Level Tokenization</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              Word-level tokenization splits text into individual words, typically using whitespace and 
              punctuation as delimiters. This is the most intuitive approach but has limitations when 
              dealing with morphologically rich languages or out-of-vocabulary (OOV) words.
            </Text>
            
            <Box mb="md">
              <BlockMath math="\text{Vocabulary Size} = |V| \approx 50,000 \text{ to } 100,000 \text{ tokens}" />
            </Box>
            
            <CodeBlock
              language="python"
              code={`
# Word-level tokenization example
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(text)
print(tokens)  # Output: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

# Calculating vocabulary size on a corpus
def get_vocabulary_size(corpus):
    all_tokens = []
    for text in corpus:
        all_tokens.extend(word_tokenize(text.lower()))
    return len(set(all_tokens))

corpus = ["The quick brown fox", "jumps over the lazy dog"]
vocab_size = get_vocabulary_size(corpus)
print(f"Vocabulary size: {vocab_size}")  # Output: Vocabulary size: 8
`}
            />
            
            <Text mt="md">
              <strong>Advantages:</strong> Intuitive, preserves word boundaries, works well for languages with clear word separators.
            </Text>
            <Text>
              <strong>Disadvantages:</strong> Large vocabulary size, can't handle OOV words, struggles with morphologically rich languages.
            </Text>
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="character-tokenization">
          <Accordion.Control>
            <Text fw={600}>Character-Level Tokenization</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              Character-level tokenization breaks text into individual characters. This approach eliminates OOV issues but 
              requires models to learn longer-range dependencies to capture word-level semantics.
            </Text>
            
            <Box mb="md">
              <BlockMath math="\text{Vocabulary Size} = |V| \approx 100 \text{ to } 1,000 \text{ tokens}" />
            </Box>
            
            <CodeBlock
              language="python"
              code={`
# Character-level tokenization
text = "Hello, world!"
char_tokens = list(text)
print(char_tokens)  # Output: ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']

# Character-level vocabulary size
def get_char_vocabulary_size(corpus):
    all_chars = []
    for text in corpus:
        all_chars.extend(list(text))
    return len(set(all_chars))

corpus = ["Hello, world!", "Python is great", "NLP is fun"]
char_vocab_size = get_char_vocabulary_size(corpus)
print(f"Character vocabulary size: {char_vocab_size}")  # Much smaller than word-level
`}
            />
            
            <Text mt="md">
              <strong>Advantages:</strong> No OOV problem, small vocabulary size, handles spelling variations.
            </Text>
            <Text>
              <strong>Disadvantages:</strong> Very long sequences, loses word-level information, computationally expensive for long texts.
            </Text>
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="subword-tokenization">
          <Accordion.Control>
            <Text fw={600}>Subword Tokenization</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              Subword tokenization strikes a balance between word and character-level approaches by breaking words into 
              meaningful subunits. Common words remain intact, while rare words are split into subword units. This approach 
              is dominant in modern NLP models like BERT, GPT, and RoBERTa.
            </Text>
            
            <Box mb="md">
              <BlockMath math="\text{Vocabulary Size} = |V| \approx 10,000 \text{ to } 50,000 \text{ tokens}" />
            </Box>
            
            <Text mb="md">
              The advantage of subword tokenization can be mathematically expressed as:
            </Text>
            
            <Box mb="md">
              <BlockMath math="\text{Compression Ratio} = \frac{\text{Tokens in Subword Representation}}{\text{Tokens in Character Representation}} \ll 1" />
            </Box>
            
            <Text mb="md">
              While still maintaining the expressivity:
            </Text>
            
            <Box mb="md">
              <BlockMath math="\text{Vocabulary Coverage} = \frac{|\text{Words representable}|}{|\text{All possible words}|} \approx 1" />
            </Box>
            
            <CodeBlock
              language="python"
              code={`
# Subword tokenization with Hugging Face
from transformers import AutoTokenizer

# Load a tokenizer that uses subword tokenization (WordPiece)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "The tokenizer splits uncommon words into subwords for better representation."
tokens = tokenizer.tokenize(text)
print(tokens)  
# Output: ['the', 'token', '##izer', 'splits', 'un', '##common', 'words', 'into', 'sub', '##words', 'for', 'better', 'representation', '.']

# Notice how "tokenizer" is split into "token" and "##izer", and "uncommon" into "un" and "##common"
# The "##" prefix indicates that this subword is a continuation of the previous token
`}
            />
            
            <Text mt="md">
              <strong>Advantages:</strong> Balances vocabulary size and expressivity, handles OOV words, effective for morphologically rich languages.
            </Text>
            <Text>
              <strong>Disadvantages:</strong> More complex to implement, may split semantic units, requires knowledge of language structure.
            </Text>
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="sentencepiece">
          <Accordion.Control>
            <Text fw={600}>SentencePiece and Unigram Tokenization</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              SentencePiece is a language-agnostic tokenizer that treats whitespace as a regular character, making it 
              suitable for languages without explicit word boundaries. It typically uses unigram language model tokenization,
              which models the probability of subword sequences.
            </Text>
            
            <Text mb="md">
              The unigram language model finds the most likely segmentation by maximizing the product of subword probabilities:
            </Text>
            
            <Box mb="md">
              <BlockMath math="P(X) = \prod_{i=1}^{N} P(x_i)" />
            </Box>
            
            <Text mb="md">
              Where <InlineMath math="X = (x_1, x_2, ..., x_N)" /> is a sequence of subwords and <InlineMath math="P(x_i)" /> is the probability of subword <InlineMath math="x_i" />.
            </Text>
            
            <CodeBlock
              language="python"
              code={`
# SentencePiece tokenization
# First, install SentencePiece: pip install sentencepiece

import sentencepiece as spm

# Training a SentencePiece model (typically done once on a large corpus)
with open('corpus.txt', 'w') as f:
    f.write("Example sentences for training SentencePiece tokenizer.\\n")
    f.write("It works well for multiple languages and treats spaces as normal characters.\\n")
    f.write("This helps with languages that don't use spaces between words.\\n")

# Train the model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='spm_model',
    vocab_size=1000,
    model_type='unigram'  # Can also use 'bpe', 'char', or 'word'
)

# Load the model
sp = spm.SentencePieceProcessor()
sp.load('spm_model.model')

# Tokenize text
text = "SentencePiece tokenizes text without assuming word boundaries."
tokens = sp.encode_as_pieces(text)
print(tokens)  
# Output might look like: ['▁Sentence', 'Piece', '▁tokenizes', '▁text', '▁without', '▁assuming', '▁word', '▁bound', 'aries', '.']
# Note the '▁' symbol indicating the beginning of a word (space)

# Convert back to text
decoded = sp.decode_pieces(tokens)
print(decoded)  # Should match the original text
`}
            />
            
            <Text mt="md">
              <strong>Advantages:</strong> Language-agnostic, handles any language without preprocessing, preserves original text reversibility.
            </Text>
            <Text>
              <strong>Disadvantages:</strong> Requires training on a corpus, may not preserve linguistic units in all cases.
            </Text>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
      
      <Text mb="md">
        The choice of tokenization approach depends on your specific NLP task, language, and available computational resources.
        Modern NLP models predominantly use subword tokenization methods like BPE, WordPiece, or Unigram models.
      </Text>

      {/* Tokenizer Comparison Section */}
      <SectionDivider />
      <Title order={2} id="tokenizers" mb="sm">Tokenizer Comparison</Title>
      
      <Text mb="md">
        Subword tokenization algorithms differ in how they segment text into subword units. Here's a comparison 
        of the most commonly used subword tokenization methods:
      </Text>
      
      <Card shadow="sm" p="md" mb="lg">
        <Table>
          <thead>
            <tr>
              <th>Algorithm</th>
              <th>Approach</th>
              <th>Models Using It</th>
              <th>Key Characteristics</th>
              <th>Implementation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Byte Pair Encoding (BPE)</strong></td>
              <td>Iterative merging of most frequent character pairs</td>
              <td>GPT, RoBERTa, XLM</td>
              <td>
                <List size="sm">
                  <List.Item>Bottom-up approach</List.Item>
                  <List.Item>Starts with characters and builds larger units</List.Item>
                  <List.Item>Deterministic segmentation</List.Item>
                </List>
              </td>
              <td>tokenizers, SentencePiece</td>
            </tr>
            <tr>
              <td><strong>WordPiece</strong></td>
              <td>Merges pairs that maximize likelihood of training data</td>
              <td>BERT, DistilBERT, Electra</td>
              <td>
                <List size="sm">
                  <List.Item>Similar to BPE but with likelihood criterion</List.Item>
                  <List.Item>Uses "##" prefix for subword continuations</List.Item>
                  <List.Item>Deterministic segmentation</List.Item>
                </List>
              </td>
              <td>Hugging Face Transformers</td>
            </tr>
            <tr>
              <td><strong>Unigram</strong></td>
              <td>Probabilistic model that optimizes likelihood</td>
              <td>T5, ALBERT, XLNet</td>
              <td>
                <List size="sm">
                  <List.Item>Top-down approach</List.Item>
                  <List.Item>Starts with full vocabulary and prunes</List.Item>
                  <List.Item>Multiple possible segmentations</List.Item>
                  <List.Item>Sentence-level optimization</List.Item>
                </List>
              </td>
              <td>SentencePiece</td>
            </tr>
          </tbody>
        </Table>
      </Card>
      
      <Text mb="md">
        Let's implement and compare these tokenization approaches on the same example:
      </Text>
      
      <CodeBlock
        language="python"
        code={`
# Comparing different tokenization methods
from transformers import AutoTokenizer
import sentencepiece as spm

# Example text
text = "Transferring learning to understand unsupervised representation learning"

# 1. BPE tokenization (GPT-2)
bpe_tokenizer = AutoTokenizer.from_pretrained("gpt2")
bpe_tokens = bpe_tokenizer.tokenize(text)
print("BPE (GPT-2):", bpe_tokens)
# Output: ['Transfer', 'ring', 'learning', 'to', 'understand', 'un', 'super', 'vised', 'representation', 'learning']

# 2. WordPiece tokenization (BERT)
wordpiece_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
wordpiece_tokens = wordpiece_tokenizer.tokenize(text)
print("WordPiece (BERT):", wordpiece_tokens)
# Output: ['transfer', '##ring', 'learning', 'to', 'understand', 'un', '##sup', '##er', '##vis', '##ed', 'representation', 'learning']

# 3. Unigram tokenization (T5)
unigram_tokenizer = AutoTokenizer.from_pretrained("t5-small")
unigram_tokens = unigram_tokenizer.tokenize(text)
print("Unigram (T5):", unigram_tokens)
# Output: ['▁Transfer', 'ring', '▁learning', '▁to', '▁understand', '▁un', 'super', 'vis', 'ed', '▁representation', '▁learning']

# Vocabulary sizes
print(f"BPE vocabulary size: {len(bpe_tokenizer.get_vocab())}")
print(f"WordPiece vocabulary size: {len(wordpiece_tokenizer.get_vocab())}")
print(f"Unigram vocabulary size: {len(unigram_tokenizer.get_vocab())}")
`}
      />
      
      <Text mt="md" mb="md">
        The mathematical formulation for the different tokenization algorithms is as follows:
      </Text>
      
      <Grid mb="lg">
        <Grid.Col span={6}>
          <Card shadow="sm" p="md">
            <Title order={4} mb="sm">BPE (Byte Pair Encoding)</Title>
            <Text mb="md">
              BPE starts with characters and iteratively merges the most frequent adjacent pairs:
            </Text>
            <Box>
              <BlockMath math="\text{arg\,max}_{pair \in Pairs} \text{count}(pair)" />
            </Box>
            <Text mb="md" mt="md">
              Where <InlineMath math="Pairs" /> is the set of all adjacent token pairs in the corpus.
            </Text>
          </Card>
        </Grid.Col>
        
        <Grid.Col span={6}>
          <Card shadow="sm" p="md">
            <Title order={4} mb="sm">Unigram Language Model</Title>
            <Text mb="md">
              Unigram finds the segmentation that maximizes the likelihood:
            </Text>
            <Box>
              <BlockMath math="S^* = \text{arg\,max}_{S} \prod_{i=1}^{|S|} P(s_i)" />
            </Box>
            <Text mt="md">
              Where <InlineMath math="S = \{s_1, s_2, ..., s_{|S|}\}" /> is a possible segmentation of the text and <InlineMath math="P(s_i)" /> is the probability of subword <InlineMath math="s_i" />.
            </Text>
          </Card>
        </Grid.Col>
      </Grid>
      
      <Text>
        The core tradeoff in tokenization is between vocabulary size and sequence length:
      </Text>
      
      <Box my="md">
        <BlockMath math="\text{Vocabulary Size} \times \text{Average Sequence Length} \approx \text{constant}" />
      </Box>
      
      <Text mb="md">
        Subword tokenization algorithms aim to find an optimal balance between these two factors.
      </Text>
      
      {/* Special Tokens Section */}
      <SectionDivider />
      <Title order={2} id="special-tokens" mb="sm">Special Tokens</Title>
      
      <Text mb="md">
        Special tokens are reserved tokens with specific meanings in NLP models. They play crucial roles 
        in defining input structure, handling sequence boundaries, and informing the model about special elements.
      </Text>
      
      <Card shadow="sm" p="md" mb="lg">
        <Table>
          <thead>
            <tr>
              <th>Special Token</th>
              <th>Symbol (Example)</th>
              <th>Purpose</th>
              <th>Models Using It</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Beginning of Sequence</td>
              <td>[BOS], &lt;s&gt;</td>
              <td>Marks the start of a sequence</td>
              <td>RoBERTa, GPT</td>
            </tr>
            <tr>
              <td>End of Sequence</td>
              <td>[EOS], &lt;/s&gt;</td>
              <td>Marks the end of a sequence</td>
              <td>BERT, GPT, T5</td>
            </tr>
            <tr>
              <td>Classification Token</td>
              <td>[CLS]</td>
              <td>Used for classification tasks, typically at the start</td>
              <td>BERT, DistilBERT</td>
            </tr>
            <tr>
              <td>Separator Token</td>
              <td>[SEP]</td>
              <td>Separates different sentences or segments</td>
              <td>BERT, RoBERTa</td>
            </tr>
            <tr>
              <td>Padding Token</td>
              <td>[PAD]</td>
              <td>Used to pad sequences to uniform length</td>
              <td>Most models</td>
            </tr>
            <tr>
              <td>Unknown Token</td>
              <td>[UNK]</td>
              <td>Represents tokens not in vocabulary</td>
              <td>Most models</td>
            </tr>
            <tr>
              <td>Mask Token</td>
              <td>[MASK]</td>
              <td>Used for masked language modeling</td>
              <td>BERT, RoBERTa</td>
            </tr>
          </tbody>
        </Table>
      </Card>
      
      <Text mb="md">
        Special tokens are crucial for model understanding. Let's see how to use them with Hugging Face Transformers:
      </Text>
      
      <CodeBlock
        language="python"
        code={`
# Using special tokens with Hugging Face
from transformers import AutoTokenizer
import torch

# Load a BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# View special tokens
print(f"CLS token: {tokenizer.cls_token}, ID: {tokenizer.cls_token_id}")
print(f"SEP token: {tokenizer.sep_token}, ID: {tokenizer.sep_token_id}")
print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
print(f"UNK token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")
print(f"All special tokens: {tokenizer.all_special_tokens}")

# Simple sequence tokenization
text = "This is an example."
tokens = tokenizer(text)
print("Token IDs:", tokens["input_ids"])
print("Token string:", tokenizer.convert_ids_to_tokens(tokens["input_ids"]))
# Notice the [CLS] at beginning and [SEP] at end

# Multiple sequences with special tokens
text_a = "How are you?"
text_b = "I am fine, thank you!"
tokens = tokenizer(text_a, text_b)
print("Multiple sequences:")
print("Token IDs:", tokens["input_ids"])
print("Token string:", tokenizer.convert_ids_to_tokens(tokens["input_ids"]))
# Output will include: [CLS] text_a [SEP] text_b [SEP]

# Masked language modeling example
text = "The capital of France is [MASK]."
tokens = tokenizer(text)
print("Masked example:")
print("Token IDs:", tokens["input_ids"])
print("Token string:", tokenizer.convert_ids_to_tokens(tokens["input_ids"]))
`}
      />
      
      <Text mt="md" mb="md">
        Different models may use different special tokens or use them in different ways. Always check 
        the documentation for your specific model to understand its token usage patterns.
      </Text>
      
      {/* Sequence Length Handling Section */}
      <SectionDivider />
      <Title order={2} id="sequence-length" mb="sm">Sequence Length Handling</Title>
      
      <Text mb="md">
        Transformer models have a maximum context length (typically 512-2048 tokens) due to the 
        quadratic complexity of self-attention. Proper handling of sequence lengths is essential for 
        optimal model performance.
      </Text>
      
      <Accordion mb="lg">
        <Accordion.Item value="truncation">
          <Accordion.Control>
            <Text fw={600}>Truncation Strategies</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              Truncation reduces sequences that exceed the model's maximum length. There are several approaches:
            </Text>
            
            <Card shadow="sm" p="md" mb="md">
              <Table>
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Description</th>
                    <th>When to Use</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Simple truncation</td>
                    <td>Cuts off tokens exceeding the limit</td>
                    <td>When later tokens are less important</td>
                  </tr>
                  <tr>
                    <td>Truncate from beginning</td>
                    <td>Removes tokens from the start</td>
                    <td>When recent tokens are more relevant (e.g., chat logs)</td>
                  </tr>
                  <tr>
                    <td>Middle truncation</td>
                    <td>Keeps beginning and end, removes middle</td>
                    <td>When both context setup and conclusion matter</td>
                  </tr>
                  <tr>
                    <td>Stride truncation</td>
                    <td>Processes text in overlapping chunks</td>
                    <td>For tasks requiring full context (e.g., QA, long text analysis)</td>
                  </tr>
                </tbody>
              </Table>
            </Card>
            
            <CodeBlock
              language="python"
              code={`
# Truncation strategies with Hugging Face
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
long_text = "This is a very long document that exceeds the maximum sequence length. " * 50

# Simple truncation (from right)
tokens_simple = tokenizer(long_text, truncation=True, max_length=10)
print("Simple truncation:", tokenizer.decode(tokens_simple["input_ids"]))

# Truncate from beginning
tokens_left = tokenizer(long_text, truncation='only_first', max_length=10)
print("Left truncation:", tokenizer.decode(tokens_left["input_ids"]))

# Stride truncation with overlap for document processing
stride = 5  # Number of overlapping tokens
max_length = 10
all_chunks = []
for i in range(0, len(long_text), stride):
    chunk = long_text[i:i+max_length+stride]
    tokens = tokenizer(chunk, truncation=True, max_length=max_length)
    all_chunks.append(tokens)
    
print(f"Document processed in {len(all_chunks)} chunks with stride {stride}")
`}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="padding">
          <Accordion.Control>
            <Text fw={600}>Padding Strategies</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              Padding is used to make all sequences in a batch the same length, which is required for efficient processing:
            </Text>
            
            <Card shadow="sm" p="md" mb="md">
              <Table>
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Description</th>
                    <th>Efficiency</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>No padding</td>
                    <td>Process each sequence individually</td>
                    <td>Inefficient for batch processing</td>
                  </tr>
                  <tr>
                    <td>Pad to max length</td>
                    <td>Pad all sequences to model's maximum length</td>
                    <td>Wastes computation on padding tokens</td>
                  </tr>
                  <tr>
                    <td>Pad to longest in batch</td>
                    <td>Pad to length of longest sequence in batch</td>
                    <td>Optimal for mixed-length sequences</td>
                  </tr>
                  <tr>
                    <td>Bucketing</td>
                    <td>Group similar-length sequences in batches</td>
                    <td>Best for datasets with varied lengths</td>
                  </tr>
                </tbody>
              </Table>
            </Card>
            
            <Text mb="md">
              The attention mask is a binary tensor that indicates which tokens should be attended to (1) and which should be ignored (0, padding):
            </Text>
            
            <Box mb="md">
              <BlockMath math="\text{mask}_{ij} = \begin{cases} 
                0 & \text{if token } j \text{ is padding} \\
                1 & \text{otherwise}
              \end{cases}" />
            </Box>
            
            <CodeBlock
              language="python"
              code={`
# Padding examples with Hugging Face
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sample sentences of different lengths
sentences = [
    "Hi.",
    "How are you?",
    "This is a longer sentence with more tokens."
]

# No padding (default)
tokens_no_pad = tokenizer(sentences)
print("No padding:", [len(ids) for ids in tokens_no_pad["input_ids"]])

# Pad to max length
tokens_max_pad = tokenizer(sentences, padding='max_length', max_length=20)
print("Max padding:", [len(ids) for ids in tokens_max_pad["input_ids"]])
print("Attention masks:", tokens_max_pad["attention_mask"])

# Pad to longest in batch
tokens_batch_pad = tokenizer(sentences, padding=True)
print("Batch padding:", [len(ids) for ids in tokens_batch_pad["input_ids"]])
print("Attention masks:", tokens_batch_pad["attention_mask"])

# Converting to PyTorch tensors for model input
inputs = tokenizer(sentences, padding=True, return_tensors="pt")
print("Input IDs shape:", inputs["input_ids"].shape)
print("Attention mask shape:", inputs["attention_mask"].shape)

# Example showing how attention mask works
token_strs = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]]
for i, (tokens, mask) in enumerate(zip(token_strs, inputs["attention_mask"])):
    attended = [t for t, m in zip(tokens, mask) if m == 1]
    ignored = [t for t, m in zip(tokens, mask) if m == 0]
    print(f"Sentence {i}:")
    print(f"  Attended tokens: {attended}")
    print(f"  Ignored tokens: {ignored}")
`}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="dynamic-processing">
          <Accordion.Control>
            <Text fw={600}>Dynamic Document Processing</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              For long documents that exceed model context windows, several techniques can be used:
            </Text>
            
            <Card shadow="sm" p="md" mb="md">
              <Table>
                <thead>
                  <tr>
                    <th>Technique</th>
                    <th>Description</th>
                    <th>Suitable Tasks</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Sliding window</td>
                    <td>Process text in overlapping chunks</td>
                    <td>Named entity recognition, text classification</td>
                  </tr>
                  <tr>
                    <td>Hierarchical processing</td>
                    <td>First process chunks, then combine results</td>
                    <td>Document classification, summarization</td>
                  </tr>
                  <tr>
                    <td>Recursive processing</td>
                    <td>Iteratively refine output based on chunks</td>
                    <td>Summarization, question answering</td>
                  </tr>
                </tbody>
              </Table>
            </Card>
            
            <CodeBlock
              language="python"
              code={`
# Sliding window approach for long document processing
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def process_long_document(document, max_length=512, stride=256):
    """
    Process a long document using a sliding window approach with an overlap (stride).
    
    Args:
        document (str): The document text to process
        max_length (int): Maximum sequence length for the model
        stride (int): Number of overlapping tokens between windows
        
    Returns:
        torch.Tensor: Document embeddings (averaged across chunks)
    """
    # Tokenize the full document
    tokens = tokenizer(document)
    input_ids = tokens["input_ids"]
    
    # If document fits in one window, process it directly
    if len(input_ids) <= max_length:
        inputs = tokenizer(document, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    
    # Process document in chunks with overlap
    all_embeddings = []
    
    # Process document in chunks
    for i in range(0, len(input_ids), max_length - stride):
        # Extract chunk (with overlap)
        chunk_ids = input_ids[i:i + max_length]
        
        # Skip very small last chunks
        if len(chunk_ids) < max_length // 2:
            continue
            
        # Convert to PyTorch tensors
        inputs = tokenizer.encode_plus(
            tokenizer.decode(chunk_ids),  # Convert back to text and re-encode to handle special tokens
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        
        # Process with model
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get embeddings (mean pooling of last hidden state)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(chunk_embedding)
    
    # Combine embeddings from all chunks (mean pooling)
    document_embedding = torch.mean(torch.cat(all_embeddings, dim=0), dim=0, keepdim=True)
    return document_embedding

# Example usage
long_document = "This is an example of a very long document that exceeds the maximum context window. " * 100
document_embedding = process_long_document(long_document)
print(f"Document embedding shape: {document_embedding.shape}")
`}
            />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
      
      <Text mb="md">
        Proper sequence length handling is crucial for model performance, especially when dealing with long documents or batched processing.
      </Text>
      
      {/* Interactive Data Panel */}
      <SectionDivider />
      <Title order={2} mb="sm">Hands-on Tokenization</Title>
      
      
      {/* Summary Section */}
      <SectionDivider />
      <Title order={2} mb="sm">Summary</Title>
      
      <Card shadow="sm" p="md" mb="lg">
        <List>
          <List.Item><strong>Text Cleaning:</strong> Different cleaning techniques serve different purposes. Modern transformer models require minimal preprocessing.</List.Item>
          <List.Item><strong>Tokenization Approaches:</strong> The three main approaches are word-level, character-level, and subword tokenization, with subword being the dominant approach in modern NLP.</List.Item>
          <List.Item><strong>Tokenization Algorithms:</strong> BPE, WordPiece, and Unigram models represent different approaches to subword tokenization, each with strengths and weaknesses.</List.Item>
          <List.Item><strong>Special Tokens:</strong> Models use special tokens like [CLS], [SEP], and [MASK] for specific purposes within the architecture.</List.Item>
          <List.Item><strong>Sequence Length:</strong> Proper handling of sequence length through truncation, padding, and chunking is essential for effective model operation.</List.Item>
        </List>
      </Card>
      
      <Text mb="md">
        Preprocessing and tokenization, while sometimes overlooked, are crucial components of any NLP pipeline. 
        The choices made at this stage can significantly impact downstream model performance.
      </Text>
    </Container>
  );
};

export default Preprocessing;