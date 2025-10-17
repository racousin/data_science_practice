import React from "react";
import { Text, Title, List, Flex, Image } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import InteractiveTokenizer from "components/InteractiveTokenizer";

const Tokenization = () => {
  return (
    <>
      <div data-slide>
        <Title order={1}>Tokenization - Represent text as numerical values</Title>

        <Text mt="md">
          Tokenization is the process of breaking text into smaller units (tokens) that serve as the basic elements
          for numerical representation. This is a fundamental preprocessing step in all NLP systems.
        </Text>

        <Text mt="md">
          Different tokenization approaches balance trade-offs between vocabulary size, semantic granularity,
          and handling of unseen words.
        </Text>
                <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/tokenize1.jpg"
            alt="Diagram showing the tokenization pipeline from text to tokens to indices and back"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
        </Flex>
      </div>


      <div data-slide>

        <Text mt="md">
          Tokenization involves three key components that work together to convert text into numerical representations:
        </Text>

        <Title order={3} mt="lg">1. Vocabulary</Title>
        <Text mt="sm">
          A fixed collection of all possible tokens that the tokenizer recognizes. Each unique token in the vocabulary is assigned a unique integer identifier (token ID).
        </Text>
        <Text mt="lg">
          The vocabulary size is a critical parameter affecting both model size and performance. Larger vocabularies require more parameters but can represent text more precisely.
        </Text>
        <Title order={3} mt="lg">2. Tokens</Title>
        <Text mt="sm">
          The individual units that text is split into. These can be words, characters, or subword pieces depending on the tokenization strategy.
        </Text>


        <Title order={3} mt="lg">3. Token IDs</Title>
        <Text mt="sm">
          Integer identifiers assigned to each token in the vocabulary. These numbers are what neural networks actually process.
        </Text>


      </div>
      <div data-slide>
        <Title order={2}>Special Tokens</Title>

        <Text mt="md">
          Most tokenizers include special tokens in their vocabulary for specific purposes:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item><strong>&lt;PAD&gt;</strong>: Padding token to make sequences uniform length</List.Item>
          <List.Item><strong>&lt;UNK&gt;</strong>: Unknown token for out-of-vocabulary words</List.Item>
          <List.Item><strong>&lt;BOS&gt;/&lt;EOS&gt;</strong>: Beginning/End of sequence markers</List.Item>
          <List.Item><strong>[CLS]/[SEP]</strong>: Classification and separator tokens (used in BERT)</List.Item>
        </List>

        <Text mt="md">
          These special tokens are included in the vocabulary and have their own unique token IDs, just like regular tokens.
        </Text>

      </div>

      <div data-slide>
        <Title order={2}>Example: Complete Tokenization Process</Title>

        <Text mt="md">
          Consider the text: "Natural language processing is fascinating!"
        </Text>

        <Title order={3} mt="lg">Split into Tokens</Title>
        <CodeBlock
          language="python"
          code={`text = "Natural language processing is fascinating!"

vocabulary = {
    "<PAD>": 0,      # Special: padding token
    "<UNK>": 1,      # Special: unknown token
    "Natural": 2,
    "language": 3,
    "processing": 4,
    "is": 5,
    "fascinating": 6,
    "!": 7
}

# Vocabulary size: 8 tokens

tokens = ["Natural", "language", "processing", "is", "fascinating", "!"]`}
        />
      </div>

      <div data-slide>

        <Title order={3} mt="md">Convert Tokens to IDs</Title>
        <CodeBlock
          language="python"
          code={`# Look up each token in the vocabulary
token_ids = [2, 3, 4, 5, 6, 7]

# These numbers are what the model processes`}
        />

        <Title order={3} mt="lg">Convert IDs Back to Tokens</Title>
        <CodeBlock
          language="python"
          code={`# Reverse lookup: IDs → tokens
decoded_tokens = ["Natural", "language", "processing", "is", "fascinating", "!"]`}
        />

        <Title order={3} mt="lg">Reconstruct Text</Title>
        <CodeBlock
          language="python"
          code={`# Join tokens to form original text
reconstructed = "Natural language processing is fascinating!"`}
        />

        <Title order={3} mt="lg">Handling Unknown Words</Title>
        <CodeBlock
          language="python"
          code={`# New text with words not in vocabulary
new_text = "Natural language understanding is amazing!"

# "understanding" and "amazing" → <UNK> (ID: 1)
new_token_ids = [2, 3, 1, 5, 1, 7]`}
        />
      </div>



      <div data-slide>
        <Title order={2}>Word-Level Tokenization</Title>

        <Text mt="md">
          Word tokenization splits text at word boundaries, typically using spaces and punctuation as delimiters.
          Each complete word becomes a single token.
        </Text>

        <Title order={3} mt="lg">Process</Title>

        <Text mt="md">
          Consider the text: "playing games"
        </Text>

        <CodeBlock
          language="python"
          code={`text = "playing games"
tokens = text.split()  # Split on spaces
print(tokens)  # ['playing', 'games']`}
        />

        <Text mt="md">
          Each word is then mapped to a unique ID from the vocabulary:
        </Text>

        <CodeBlock
          language="python"
          code={`vocabulary = {"playing": 1523, "games": 2847}
token_ids = [1523, 2847]`}
        />

        <Text mt="md">
          If a word is not in the vocabulary, it becomes an unknown token:
        </Text>

        <CodeBlock
          language="python"
          code={`text = "playing videogames"  # "videogames" not in vocab
tokens = ["playing", "<UNK>"]
token_ids = [1523, 0]  # <UNK> mapped to ID 0`}
        />

        <Title order={3} mt="lg">Trade-offs</Title>

        <Text mt="md" fw={500}>Advantages:</Text>
        <List spacing="xs" mt="sm">
          <List.Item>Preserves semantic meaning - each token is a complete word</List.Item>
          <List.Item>Intuitive and easy to interpret</List.Item>
        </List>

        <Text mt="md" fw={500}>Disadvantages:</Text>
        <List spacing="xs" mt="sm">
          <List.Item>Large vocabulary required (tens of thousands of words)</List.Item>
          <List.Item>Cannot handle unknown words - they become &lt;UNK&gt;</List.Item>
          <List.Item>Cannot capture morphological relationships (play, playing, played are separate)</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>Character-Level Tokenization</Title>

        <Text mt="md">
          Character tokenization breaks text into individual characters. This creates very long sequences
          but uses a minimal vocabulary.
        </Text>

        <Title order={3} mt="lg">Process</Title>

        <CodeBlock
          language="python"
          code={`text = "playing games"
tokens = list(text)  # Split into characters
print(tokens)  # ['p','l','a','y','i','n','g',' ','g','a','m','e','s']`}
        />

        <Text mt="md">
          Each character is mapped to a unique ID:
        </Text>

        <CodeBlock
          language="python"
          code={`# Build vocabulary from unique characters
vocab = {' ': 0, 'a': 1, 'e': 2, 'g': 3, 'i': 4, 'l': 5,
         'm': 6, 'n': 7, 'p': 8, 's': 9, 'y': 10}
token_ids = [8,5,1,10,4,7,3,0,3,1,6,2,9]  # 13 tokens`}
        />

        <Text mt="md">
          Character-level tokenization never encounters unknown characters if the vocabulary includes
          all possible characters:
        </Text>

        <CodeBlock
          language="python"
          code={`text = "playing videogames"
# All characters are in vocabulary - no <UNK> needed
tokens = list(text)  # 18 tokens total`}
        />

        <Title order={3} mt="lg">Trade-offs</Title>

        <Text mt="md" fw={500}>Advantages:</Text>
        <List spacing="xs" mt="sm">
          <List.Item>Very small vocabulary (typically less than 100 tokens)</List.Item>
          <List.Item>No unknown tokens - handles any text</List.Item>
          <List.Item>Works well for languages without clear word boundaries</List.Item>
        </List>

        <Text mt="md" fw={500}>Disadvantages:</Text>
        <List spacing="xs" mt="sm">
          <List.Item>Very long sequences (each character is a token)</List.Item>
          <List.Item>Loses semantic information - individual characters have no meaning</List.Item>
          <List.Item>Model must learn to compose characters into meaningful units</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>Subword Tokenization</Title>

        <Text mt="md">
          Subword tokenization methods break words into meaningful subword units, balancing vocabulary size
          against semantic granularity.
        </Text>

        <Text mt="md">
          Three popular algorithms:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item><strong>Byte-Pair Encoding (BPE)</strong></List.Item>
          <List.Item><strong>WordPiece</strong></List.Item>
          <List.Item><strong>SentencePiece</strong></List.Item>
        </List>

        <Text mt="lg">
          These methods enable models to handle out-of-vocabulary words by decomposing them into known subword units.
        </Text>

      </div>

      <div data-slide>
        <Title order={2}>N-grams</Title>

        <Text mt="md">
          N-grams are sequences of N consecutive tokens.
        </Text>


                <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/N-grams.webp"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
        </Flex>

      </div>

      <div data-slide>
        <Title order={2}>Byte-Pair Encoding (BPE)</Title>

        <Text mt="md">
          BPE is a subword tokenization method that starts with characters and iteratively merges
          the most frequent adjacent pairs (2-grams) to build larger tokens. This creates a vocabulary between
          character-level and word-level.
        </Text>

        <Title order={3} mt="lg">Training Corpus</Title>

        <CodeBlock
          language="python"
          code={`# Training data
corpus = ["low", "low", "low", "lowest", "lowest"]

# Initial: split into characters
# "l o w", "l o w", "l o w", "l o w e s t", "l o w e s t"`}
        />

        <Text mt="md">
          Initial vocabulary: ['l', 'o', 'w', 'e', 's', 't']
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>BPE: Iterative Merging</Title>

        <Text mt="md">
          Step 1: Count all adjacent character pairs
        </Text>

        <CodeBlock
          language="python"
          code={`# Pair frequencies:
# ('l','o'): 5 times  ← most frequent
# ('o','w'): 5 times  ← most frequent
# ('w','e'): 2 times
# ('e','s'): 2 times
# ('s','t'): 2 times`}
        />

        <Text mt="md">
          Step 2: Merge most frequent pair ('l','o') → 'lo'
        </Text>

        <CodeBlock
          language="python"
          code={`# After merge:
# "lo w", "lo w", "lo w", "lo w e s t", "lo w e s t"
# Vocabulary: ['l', 'o', 'w', 'e', 's', 't', 'lo']`}
        />

        <Text mt="md">
          Step 3: Continue merging ('lo','w') → 'low'
        </Text>

        <CodeBlock
          language="python"
          code={`# After merge:
# "low", "low", "low", "low e s t", "low e s t"
# Vocabulary: ['l', 'o', 'w', 'e', 's', 't', 'lo', 'low']`}
        />
      </div>

      <div data-slide>
        <Title order={2}>BPE: Tokenizing New Words</Title>

        <Text mt="md">
          After training, BPE can tokenize any word using learned subwords:
        </Text>

        <CodeBlock
          language="python"
          code={`# Vocabulary learned: ['l', 'o', 'w', 'e', 's', 't', 'lo', 'low']

# Tokenize "low"
tokens = ["low"]  # Found as single token`}
        />

        <CodeBlock
          language="python"
          code={`# Tokenize "lower" (not in training data)
tokens = ["low", "e", "r"]  # "low" found, rest as characters`}
        />

        <Title order={3} mt="lg">Trade-offs</Title>

        <Text mt="md" fw={500}>Advantages:</Text>
        <List spacing="xs" mt="sm">
          <List.Item>Balanced vocabulary size (moderate, not too large or small)</List.Item>
          <List.Item>Handles unknown words by breaking them into subwords</List.Item>
          <List.Item>Captures morphology (low, lower, lowest share "low")</List.Item>
          <List.Item>Most widely used in modern NLP (GPT, BERT, etc.)</List.Item>
        </List>

        <Text mt="md" fw={500}>Disadvantages:</Text>
        <List spacing="xs" mt="sm">
          <List.Item>Requires training on a corpus to learn merge operations</List.Item>
          <List.Item>More complex than word or character tokenization</List.Item>
          <List.Item>Tokenization depends on the training data distribution</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>Real-World BPE Tokenizer Examples</Title>

        <Title order={3} mt="lg">GPT-2 Tokenizer (2019)</Title>

        <List spacing="sm" mt="sm">
          <List.Item><strong>Vocabulary Size:</strong> 50,257 tokens</List.Item>
          <List.Item><strong>Training Corpus:</strong> WebText - 8 million web pages from Reddit outbound links</List.Item>
          <List.Item><strong>Merge Rules:</strong> BPE merge operations</List.Item>
        </List>

        <Text mt="md">
          Source: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
        </Text>
      </div>

      <div data-slide>

        <Title order={3} mt="lg">Llama 3 Tokenizer (2024)</Title>

        <List spacing="sm" mt="sm">
          <List.Item><strong>Vocabulary Size:</strong> 128,256 tokens</List.Item>
            <List.Item><strong>Training Corpus:</strong> Multilingual data including non-English languages</List.Item>
          <List.Item><strong>Algorithm:</strong> tiktoken-based BPE</List.Item>

        </List>


        <Text mt="md">
          Source: "The Llama 3 Herd of Models" (Meta AI, 2024)
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Interactive Tokenization</Title>

        <Text mt="md">
          Experiment with GPT tokenization by entering text below. Each colored badge represents a single token.
        </Text>

        <InteractiveTokenizer />

        <Text mt="lg" size="sm">
          Notice how the tokenizer handles:
        </Text>
        <List spacing="xs" mt="sm" size="sm">
          <List.Item>Spaces and punctuation as separate or combined tokens</List.Item>
          <List.Item>Common words as single tokens</List.Item>
          <List.Item>Uncommon or made-up words split into subword units</List.Item>
        </List>
      </div>

    </>
  );
};

export default Tokenization;
