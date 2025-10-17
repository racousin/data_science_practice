import React from "react";
import { Text, Title, List, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";

const Tokenization = () => {
  return (
    <>
      <div data-slide>
        <Title order={1}>Tokenization</Title>

        <Text mt="md">
          Tokenization is the process of breaking text into smaller units (tokens) that serve as the basic elements
          for numerical representation. This is a fundamental preprocessing step in all NLP systems.
        </Text>

        <Text mt="md">
          Different tokenization approaches balance trade-offs between vocabulary size, semantic granularity,
          and handling of unseen words.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Challenges in Text Representation</Title>

        <List spacing="sm" mt="md">
          <List.Item>
            <strong>High dimensionality:</strong> Human languages contain tens or hundreds of thousands of words,
            making direct one-hot encoding impractical
          </List.Item>
          <List.Item>
            <strong>Variable length:</strong> Unlike fixed-dimensional data like images, text inputs vary greatly in length
          </List.Item>
          <List.Item>
            <strong>Context dependency:</strong> The meaning of words changes based on surrounding context
          </List.Item>
          <List.Item>
            <strong>Morphological variation:</strong> Words appear in different forms (plurals, tenses, etc.)
            while maintaining related meanings
          </List.Item>
          <List.Item>
            <strong>Out-of-vocabulary words:</strong> New or rare words not seen during training pose representation challenges
          </List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/text-representation-challenges.png"
            alt="Visualization of text representation challenges including OOV words and morphological variation"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Key challenges in representing text: variable length, context dependency, and vocabulary issues
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Formal Components of Tokenization</Title>

        <Title order={3} mt="md">Tokenization Function</Title>
        <Text mt="sm">
          A function <InlineMath math="T" /> that maps text strings to sequences of tokens:
        </Text>
        <BlockMath math="T: \mathcal{S} \rightarrow \mathcal{V}^*" />
        <Text>
          Where <InlineMath math="\mathcal{S}" /> is the space of all text strings and <InlineMath math="\mathcal{V}" /> is the token vocabulary.
        </Text>

        <Title order={3} mt="lg">Vocabulary</Title>
        <Text mt="sm">
          A finite set <InlineMath math="\mathcal{V} = \{t_1, t_2, ..., t_{|\mathcal{V}|}\}" /> containing all possible tokens.
        </Text>
        <Text mt="sm">
          The vocabulary size <InlineMath math="|\mathcal{V}|" /> is a critical hyperparameter affecting model size and performance.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Token Encoding and Decoding</Title>

        <Title order={3} mt="md">Token Encoder</Title>
        <Text mt="sm">
          A function <InlineMath math="E" /> that maps tokens to integer indices:
        </Text>
        <BlockMath math="E: \mathcal{V} \rightarrow \{1, 2, ..., |\mathcal{V}|\}" />

        <Title order={3} mt="lg">Token Decoder</Title>
        <Text mt="sm">
          A function <InlineMath math="D" /> that maps integer indices back to tokens:
        </Text>
        <BlockMath math="D: \{1, 2, ..., |\mathcal{V}|\} \rightarrow \mathcal{V}" />
        <Text mt="sm">
          Where <InlineMath math="D = E^{-1}" />, the inverse of the encoder.
        </Text>

        <Title order={3} mt="lg">Reconstructor</Title>
        <Text mt="sm">
          A function <InlineMath math="R" /> that maps token sequences back to text:
        </Text>
        <BlockMath math="R: \mathcal{V}^* \rightarrow \mathcal{S}" />
        <Text mt="sm">
          In an ideal tokenizer, <InlineMath math="R \circ T" /> is the identity function, ensuring lossless reconstruction.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/tokenization-pipeline.png"
            alt="Diagram showing the tokenization pipeline from text to tokens to indices and back"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Complete tokenization pipeline: T (tokenize) → E (encode) → D (decode) → R (reconstruct)
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Special Tokens</Title>

        <Text mt="md">
          Most tokenizers include special tokens in their vocabulary:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item><strong>[UNK] or &lt;unk&gt;</strong>: Unknown token for out-of-vocabulary items</List.Item>
          <List.Item><strong>[PAD] or &lt;pad&gt;</strong>: Padding token to make sequences uniform length</List.Item>
          <List.Item><strong>[BOS]/[EOS] or &lt;s&gt;/&lt;/s&gt;</strong>: Beginning/End of sequence markers</List.Item>
          <List.Item><strong>[CLS]/[SEP]</strong>: Classification and separator tokens for models like BERT</List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/special-tokens-examples.png"
            alt="Examples of sequences with special tokens for different models"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Special token usage in BERT and GPT models
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Simple Tokenization Example</Title>

        <CodeBlock
          language="python"
          code={`# Example text
text = "Natural language processing is fascinating!"

# 1. Tokenization (word-level)
tokens = ["Natural", "language", "processing", "is", "fascinating", "!"]`}
        />

        <CodeBlock
          language="python"
          code={`# 2. Vocabulary creation
vocabulary = {
    "<PAD>": 0,  # Special padding token
    "<UNK>": 1,  # Unknown token
    "Natural": 2,
    "language": 3,
    "processing": 4,
    "is": 5,
    "fascinating": 6,
    "!": 7
}`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Token Encoding and Reconstruction</Title>

        <CodeBlock
          language="python"
          code={`# 3. Token encoding (converting to indices)
encoded = [2, 3, 4, 5, 6, 7]

# 4. Token decoding (converting back to tokens)
decoded = ["Natural", "language", "processing", "is", "fascinating", "!"]`}
        />

        <CodeBlock
          language="python"
          code={`# 5. Reconstruction (joining tokens back to text)
reconstructed = "Natural language processing is fascinating!"

# Handling out-of-vocabulary words
new_text = "Natural language understanding is amazing!"
# "understanding" and "amazing" are not in vocabulary, so they become <UNK>
new_encoded = [2, 3, 1, 5, 1, 7]  # Using <UNK> (index 1) for OOV words`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Word-Level Tokenization</Title>

        <Text mt="md">
          Word tokenization splits text at word boundaries, typically using spaces and punctuation as delimiters.
        </Text>

        <Title order={3} mt="lg">Formal Representation</Title>
        <Text mt="sm">
          Given a vocabulary <InlineMath math="V = \{w_1, w_2, ..., w_{|V|}\}" /> of unique words:
        </Text>
        <BlockMath math="\text{Tokenize}_{word}: \mathcal{S} \rightarrow V^*" />

        <Text mt="md">
          <strong>Dimension:</strong> For a vocabulary of size <InlineMath math="|V|" />, each token is represented
          as an index in the range <InlineMath math="\{1, 2, ..., |V|\}" />.
        </Text>

        <Text mt="md">
          <strong>Typical vocabulary size:</strong> 10,000 - 100,000 tokens
        </Text>

        <Text mt="md">
          <strong>Reconstruction:</strong> Text can be reconstructed by joining tokens with spaces.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/word-level-tokenization.png"
            alt="Example of word-level tokenization showing vocabulary and encoding"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Word-level tokenization with vocabulary mapping and OOV handling
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Character-Level Tokenization</Title>

        <Text mt="md">
          Character tokenization breaks text into individual characters, offering a very small vocabulary
          but requiring longer sequences.
        </Text>

        <Title order={3} mt="lg">Formal Representation</Title>
        <Text mt="sm">
          Given a character set <InlineMath math="\mathcal{C} = \{c_1, c_2, ..., c_{|\mathcal{C}|}\}" />:
        </Text>
        <BlockMath math="\text{Tokenize}_{char}: \mathcal{S} \rightarrow \mathcal{C}^*" />

        <Text mt="md">
          <strong>Dimension:</strong> For a character set of size <InlineMath math="|\mathcal{C}|" /> (typically 26-128),
          each character is represented as an index in <InlineMath math="\{1, 2, ..., |\mathcal{C}|\}" />.
        </Text>

        <Text mt="md">
          <strong>Typical vocabulary size:</strong> 26-256 tokens
        </Text>

        <Text mt="md">
          <strong>Reconstruction:</strong> Text can be perfectly reconstructed by joining characters with no separator.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/character-level-tokenization.png"
            alt="Example of character-level tokenization showing individual character encoding"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Character-level tokenization: small vocabulary, long sequences
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Character Tokenization Example</Title>

        <CodeBlock
          language="python"
          code={`# Character tokenization
text = "Hello World!"

# Tokenization (text → character tokens)
char_tokens = list(text)
print("Character tokens:", char_tokens)
# Output: ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!']`}
        />

        <CodeBlock
          language="python"
          code={`# Create character vocabulary and map to indices
char_vocab = sorted(set(char_tokens))
char_to_idx = {char: idx for idx, char in enumerate(char_vocab, 1)}

print("Vocabulary size:", len(char_vocab))
# Output: Vocabulary size: 9

# Convert to numerical representation
numerical_chars = [char_to_idx[char] for char in char_tokens]
print("Numerical representation:", numerical_chars)
# Output: [3, 6, 7, 7, 8, 1, 4, 8, 9, 7, 5, 2]`}
        />
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

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/tokenization-comparison.png"
            alt="Comparison of word-level, character-level, and subword tokenization approaches"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Comparison: word-level vs character-level vs subword tokenization
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Byte-Pair Encoding (BPE)</Title>

        <Text mt="md">
          BPE iteratively merges the most frequent adjacent character pairs or byte pairs to form new subword tokens.
        </Text>

        <Title order={3} mt="lg">Formal Representation</Title>
        <Text mt="sm">
          Given an initial character vocabulary <InlineMath math="\mathcal{C}" /> and a target vocabulary size <InlineMath math="k" />:
        </Text>
        <BlockMath math="\text{BPE}: \mathcal{C}^* \times \mathbb{N} \rightarrow \mathcal{V}^*" />

        <Text mt="md">
          Where <InlineMath math="\mathcal{V}" /> is the learned vocabulary with <InlineMath math="|\mathcal{V}| = |\mathcal{C}| + k" /> tokens
          after <InlineMath math="k" /> merge operations.
        </Text>

        <Text mt="md">
          <strong>Typical vocabulary size:</strong> 30,000 - 50,000 tokens
        </Text>

        <Text mt="sm" size="sm" fs="italic">
          Reference: Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016) - https://arxiv.org/abs/1508.07909
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>BPE Algorithm</Title>

        <List ordered spacing="sm" mt="md">
          <List.Item>Start with a vocabulary of individual characters</List.Item>
          <List.Item>Count frequencies of adjacent character pairs in the corpus</List.Item>
          <List.Item>Merge the most frequent pair to create a new token</List.Item>
          <List.Item>Update frequencies with the new token</List.Item>
          <List.Item>Repeat steps 2-4 until reaching desired vocabulary size or frequency threshold</List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/bpe-algorithm-flow.png"
            alt="Flowchart of the BPE algorithm showing iterative merging process"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            BPE algorithm: iterative pair frequency analysis and merging
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>BPE Example: Initial State</Title>

        <Text mt="md">
          Consider a tiny corpus: ["low", "lower", "lowest", "newer", "wider"]
        </Text>

        <Text mt="lg"><strong>Initial state:</strong></Text>
        <Text mt="sm">
          Initial vocabulary: ['l', 'o', 'w', 'e', 'r', 's', 't', 'n', 'i', 'd'] (10 tokens)
        </Text>
        <Text mt="sm">
          Initial segmentation:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>"l o w" (3 tokens)</List.Item>
          <List.Item>"l o w e r" (5 tokens)</List.Item>
          <List.Item>"l o w e s t" (6 tokens)</List.Item>
          <List.Item>"n e w e r" (5 tokens)</List.Item>
          <List.Item>"w i d e r" (5 tokens)</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>BPE Example: Iterations</Title>

        <Text mt="md"><strong>Iteration 1:</strong></Text>
        <Text mt="sm">Frequency count: ('e', 'r') appears 3 times</Text>
        <Text mt="sm">Merge 'e' + 'r' → 'er'</Text>
        <Text mt="sm">Updated vocabulary: [..., 'er'] (11 tokens)</Text>

        <Text mt="lg"><strong>Iteration 2:</strong></Text>
        <Text mt="sm">Frequency count: ('l', 'o') appears 3 times</Text>
        <Text mt="sm">Merge 'l' + 'o' → 'lo'</Text>
        <Text mt="sm">Updated vocabulary: [..., 'er', 'lo'] (12 tokens)</Text>

        <Text mt="lg"><strong>Iteration 3:</strong></Text>
        <Text mt="sm">Frequency count: ('lo', 'w') appears 3 times</Text>
        <Text mt="sm">Merge 'lo' + 'w' → 'low'</Text>
        <Text mt="sm">Updated vocabulary: [..., 'er', 'lo', 'low'] (13 tokens)</Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/bpe-merge-tree.png"
            alt="Visual representation of BPE merge iterations forming a tree structure"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            BPE merge tree: progressive token formation from characters
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>BPE in Practice</Title>

        <CodeBlock
          language="python"
          code={`from transformers import GPT2Tokenizer

# Load a pre-trained BPE tokenizer (GPT-2 uses BPE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Natural language processing is powerful."`}
        />

        <CodeBlock
          language="python"
          code={`# Tokenization (text → BPE tokens)
tokens = tokenizer.tokenize(text)
print("BPE tokens:", tokens)
# Output: ['Natural', ' language', ' processing', ' is', ' powerful', '.']

# Convert to token IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)
# Output: [8241, 2061, 4915, 318, 5411, 13]`}
        />

        <CodeBlock
          language="python"
          code={`# Vocabulary size
print("Vocabulary size:", len(tokenizer))
# Output: Vocabulary size: 50257

# Reconstruction (token IDs → text)
reconstructed_text = tokenizer.decode(token_ids)
print("Reconstructed text:", reconstructed_text)
# Output: Natural language processing is powerful.`}
        />
      </div>

      <div data-slide>
        <Title order={2}>WordPiece</Title>

        <Text mt="md">
          WordPiece is similar to BPE but uses a likelihood-based criterion for merging tokens instead of frequency.
          It marks subword units that don't begin words with '##' to aid in reconstruction.
        </Text>

        <Title order={3} mt="lg">Merge Criterion</Title>
        <Text mt="sm">
          WordPiece selects the merge that maximizes:
        </Text>
        <BlockMath math="\text{score}(x,y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}" />
        <Text mt="sm">
          This likelihood ratio prefers merges where the combined token <InlineMath math="xy" /> appears more frequently
          than would be expected if <InlineMath math="x" /> and <InlineMath math="y" /> were independent.
        </Text>

        <Text mt="md">
          <strong>Typical vocabulary size:</strong> ~30,000 tokens
        </Text>

        <Text mt="sm" size="sm" fs="italic">
          Reference: Schuster & Nakajima, "Japanese and Korean Voice Search" (2012) - https://research.google/pubs/pub37842/
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/bpe-vs-wordpiece.png"
            alt="Comparison between BPE and WordPiece merge criteria"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            BPE vs WordPiece: frequency-based vs likelihood-based merging
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>WordPiece Example</Title>

        <CodeBlock
          language="python"
          code={`from transformers import BertTokenizer

# Load a pre-trained WordPiece tokenizer (BERT uses WordPiece)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "Natural language processing uses wordpiece tokenization."`}
        />

        <CodeBlock
          language="python"
          code={`# Tokenization (text → WordPiece tokens)
tokens = tokenizer.tokenize(text)
print("WordPiece tokens:", tokens)
# Output: ['natural', 'language', 'processing', 'uses', 'word', '##piece', 'token', '##ization', '.']

# Notice how "wordpiece" is split into "word" and "##piece"
# The "##" prefix indicates this is a continuation of the previous token`}
        />

        <CodeBlock
          language="python"
          code={`# Convert to token IDs (includes [CLS] and [SEP] tokens)
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)
# Output: [101, 2269, 2653, 6254, 2224, 2773, 12195, 2106, 10104, 1012, 102]

# Vocabulary size
print("Vocabulary size:", len(tokenizer))
# Output: Vocabulary size: 30522

# Reconstruction
reconstructed_text = tokenizer.decode(token_ids)
print("Reconstructed text:", reconstructed_text)
# Output: natural language processing uses wordpiece tokenization.`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Subword Markers</Title>

        <Text mt="md">
          Different tokenizers use distinctive markers to indicate token positions within words:
        </Text>

        <List spacing="sm" mt="lg">
          <List.Item>
            <strong>WordPiece:</strong> Uses "##" prefix to mark subword tokens that continue a word
            <br />Example: "tokenization" → ["token", "##ization"]
          </List.Item>
          <List.Item>
            <strong>SentencePiece:</strong> Uses "▁" (underscore) prefix to mark tokens that begin words
            <br />Example: "tokenization" → ["▁token", "ization"]
          </List.Item>
          <List.Item>
            <strong>BPE:</strong> Typically doesn't use explicit markers in the original formulation
            <br />(though implementations vary)
          </List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/subword-markers-comparison.png"
            alt="Comparison of subword markers used by different tokenizers"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Different marker conventions: WordPiece (##), SentencePiece (▁), and BPE
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Tokenization Trade-offs</Title>

        <Title order={3} mt="md">Word-Level</Title>
        <List spacing="xs" mt="sm">
          <List.Item><strong>Pros:</strong> Preserves semantic units, intuitive</List.Item>
          <List.Item><strong>Cons:</strong> Large vocabulary, OOV problem, morphological variation</List.Item>
        </List>

        <Title order={3} mt="lg">Character-Level</Title>
        <List spacing="xs" mt="sm">
          <List.Item><strong>Pros:</strong> Small vocabulary, no OOV problem</List.Item>
          <List.Item><strong>Cons:</strong> Very long sequences, loses semantic information</List.Item>
        </List>

        <Title order={3} mt="lg">Subword-Level</Title>
        <List spacing="xs" mt="sm">
          <List.Item><strong>Pros:</strong> Balanced vocabulary size, handles OOV words, captures morphology</List.Item>
          <List.Item><strong>Cons:</strong> Requires training, adds complexity</List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/tokenization-tradeoffs.png"
            alt="Comparison table of tokenization approaches showing vocabulary size vs sequence length tradeoffs"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Tokenization strategy tradeoffs: vocabulary size, sequence length, and semantic granularity
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Practical Considerations</Title>

        <Title order={3} mt="md">Vocabulary Size Impact</Title>
        <Text mt="sm">
          Vocabulary size <InlineMath math="|\mathcal{V}|" /> directly affects:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><strong>Model size:</strong> Embedding layer has <InlineMath math="|\mathcal{V}| \times d" /> parameters</List.Item>
          <List.Item><strong>Sequence length:</strong> Smaller vocabulary → longer sequences</List.Item>
          <List.Item><strong>Training efficiency:</strong> Larger vocabulary → more parameters to train</List.Item>
          <List.Item><strong>Generalization:</strong> Larger vocabulary → more data needed for rare tokens</List.Item>
        </List>

        <Text mt="lg">
          Modern models typically use subword tokenization with vocabularies of 30K-50K tokens,
          balancing expressiveness and efficiency.
        </Text>
      </div>
    </>
  );
};

export default Tokenization;
