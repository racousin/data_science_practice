import React from "react";
import { Container, Title, Text, Card, Divider, List, Table, Image } from "@mantine/core";
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";
import LearnedEmbeddings from "./LearnedEmbeddings";

const TextNumericalRepresentation = () => {
  return (
    <Container fluid>
      <Title order={1} id="numerical-representation" mb="md">
        Numerical Representation of Text
      </Title>
      
      <Card shadow="sm" p="md" mb="lg" withBorder>
        <Text>
          Transforming human language into machine-understandable numerical formats presents several unique challenges:
        </Text>
        
        <List mt="md" spacing="sm">
          <List.Item>
            <strong>High dimensionality</strong>: Human languages contain tens or hundreds of thousands of words, making direct 
            one-hot encoding impractical.
          </List.Item>
          <List.Item>
            <strong>Variable length</strong>: Unlike fixed-dimensional data like images, text inputs vary greatly in length.
          </List.Item>
          <List.Item>
            <strong>Semantic relationships</strong>: Words have complex semantic relationships that should be preserved in numerical form.
          </List.Item>
          <List.Item>
            <strong>Context dependency</strong>: The meaning of words changes based on surrounding context.
          </List.Item>
          <List.Item>
            <strong>Morphological variation</strong>: Words appear in different forms (plurals, tenses, etc.) while maintaining related meanings.
          </List.Item>
          <List.Item>
            <strong>Out-of-vocabulary words</strong>: New or rare words not seen during training pose representation challenges.
          </List.Item>
        </List>
      </Card>

      <Text mb="lg">
        Addressing these challenges requires sophisticated techniques to transform text into numerical vectors 
        that preserve semantic relationships and enable effective machine learning. Two key components of this process
        are tokenization and embedding.
      </Text>
{/* Section 2: Tokenization */}
<Title order={2} id="tokenization" mb="sm">
  Tokenization
</Title>

<Text mb="md">
  Tokenization is the process of breaking text into smaller units (tokens) that serve as the basic elements
  for numerical representation. Different tokenization approaches balance the trade-offs between vocabulary size,
  semantic granularity, and handling of unseen words.
</Text>


<Card shadow="sm" p="md" mb="lg" withBorder>
  <Title order={4} mb="sm">Formal Components</Title>
  
  <Text mb="sm"><strong>1. Tokenization Function</strong></Text>
  <Text mb="md">
    A function <InlineMath math="T" /> that maps text strings to sequences of tokens:
    <BlockMath math="T: \mathcal{S} \rightarrow \mathcal{V}^*" />
    Where <InlineMath math="\mathcal{S}" /> is the space of all text strings and <InlineMath math="\mathcal{V}" /> is the token vocabulary.
  </Text>
  
  <Text mb="sm"><strong>2. Vocabulary</strong></Text>
  <Text mb="md">
    A finite set <InlineMath math="\mathcal{V} = \{t_1, t_2, ..., t_{|\mathcal{V}|}\}" /> containing all possible tokens.
    The vocabulary size <InlineMath math="|\mathcal{V}|" /> is a critical hyperparameter affecting model size and performance.
  </Text>
  
  <Text mb="sm"><strong>3. Token Encoder</strong></Text>
  <Text mb="md">
    A function <InlineMath math="E" /> that maps tokens to integer indices:
    <BlockMath math="E: \mathcal{V} \rightarrow \{1, 2, ..., |\mathcal{V}|\}" />
    This enables efficient lookup and processing.
  </Text>
  
  <Text mb="sm"><strong>4. Token Decoder</strong></Text>
  <Text mb="md">
    A function <InlineMath math="D" /> that maps integer indices back to tokens:
    <BlockMath math="D: \{1, 2, ..., |\mathcal{V}|\} \rightarrow \mathcal{V}" />
    Where <InlineMath math="D = E^{-1}" />, the inverse of the encoder.
  </Text>
  
  <Text mb="sm"><strong>5. Reconstructor</strong></Text>
  <Text mb="md">
    A function <InlineMath math="R" /> that maps token sequences back to text:
    <BlockMath math="R: \mathcal{V}^* \rightarrow \mathcal{S}" />
    In an ideal tokenizer, <InlineMath math="R \circ T" /> is the identity function, ensuring lossless reconstruction.
  </Text>

</Card>

<Card shadow="sm" p="md" mb="lg" withBorder>
  <Title order={4} mb="sm">Common Considerations</Title>
  
  <Text mb="sm"><strong>Text Normalization</strong></Text>
  <Text mb="md">
    Most tokenizers perform some form of text normalization, which may include:
    <ul>
      <li><strong>Case handling</strong>: Converting to lowercase, preserving case, or creating separate tokens for different cases</li>
      <li><strong>Whitespace handling</strong>: Removing excess whitespace, normalizing spaces</li>
      <li><strong>Punctuation</strong>: Separating punctuation from words, treating some punctuation as distinct tokens</li>
      <li><strong>Special characters</strong>: Handling non-alphanumeric characters</li>
    </ul>
  </Text>
  
  <Text mb="sm"><strong>Special Tokens</strong></Text>
  <Text mb="md">
    Most tokenizers include special tokens in their vocabulary:
    <ul>
      <li><strong>[UNK] or &lt;unk&gt;</strong>: Unknown token for out-of-vocabulary items</li>
      <li><strong>[PAD] or &lt;pad&gt;</strong>: Padding token to make sequences uniform length</li>
      <li><strong>[BOS]/[EOS] or &lt;s&gt;/&lt;/s&gt;</strong>: Beginning/End of sequence markers</li>
      <li><strong>[CLS]/[SEP]</strong>: Classification and separator tokens for models like BERT</li>
    </ul>
  </Text>
  
  <Text mb="sm"><strong>Subword Markers</strong></Text>
  <Text mb="md">
    Different tokenizers use distinctive markers to indicate token positions within words:
    <ul>
      <li><strong>WordPiece</strong>: Uses "##" prefix to mark subword tokens that continue a word</li>
      <li><strong>SentencePiece</strong>: Uses "▁" (underscore) prefix to mark tokens that begin words</li>
      <li><strong>BPE</strong>: Typically doesn't use explicit markers in the original formulation, though implementations vary</li>
    </ul>
  </Text>
</Card>


<Title order={4} mt="md" mb="sm">Simple Tokenization Example</Title>
  <CodeBlock
    language="python"
    code={`# Example text
text = "Natural language processing is fascinating!"

# 1. Tokenization (word-level)
tokens = ["Natural", "language", "processing", "is", "fascinating", "!"]

# 2. Vocabulary creation
vocabulary = {
    "<PAD>": 0,  # Special padding token
    "<UNK>": 1,  # Unknown token
    "Natural": 2,
    "language": 3,
    "processing": 4,
    "is": 5,
    "fascinating": 6,
    "!": 7
}

# 3. Token encoding (converting to indices)
encoded = [2, 3, 4, 5, 6, 7]  # [Natural, language, processing, is, fascinating, !]

# 4. Token decoding (converting back to tokens)
decoded = ["Natural", "language", "processing", "is", "fascinating", "!"]

# 5. Reconstruction (joining tokens back to text)
reconstructed = "Natural language processing is fascinating!"

# Handling out-of-vocabulary words
new_text = "Natural language understanding is amazing!"
new_tokens = ["Natural", "language", "understanding", "is", "amazing", "!"]
# "understanding" and "amazing" are not in vocabulary, so they become <UNK>
new_encoded = [2, 3, 1, 5, 1, 7]  # Using <UNK> (index 1) for OOV words`}
  />

<Title order={3} id="word-tokenization" mb="sm">
  Word-Level Tokenization
</Title>

<Text mb="md">
  Word tokenization splits text at word boundaries, typically using spaces and punctuation as delimiters.
</Text>

<Card shadow="sm" p="md" mb="md" withBorder>
  <Title order={4} mb="sm">Formal Representation</Title>
  <Text mb="md">
    Given a vocabulary <InlineMath math="V = \{w_1, w_2, ..., w_{|V|}\}" /> of unique words:
  </Text>
  <BlockMath math="\text{Tokenize}_{word}: \mathcal{S} \rightarrow V^*" />
  <Text mb="md">
    Where <InlineMath math="\mathcal{S}" /> is the space of all strings and <InlineMath math="V^*" /> represents sequences of words from vocabulary <InlineMath math="V" />.
  </Text>
  <Text mb="md">
    <strong>Dimension</strong>: For a vocabulary of size <InlineMath math="|V|" />, each token is represented as an index in the range <InlineMath math="\{1, 2, ..., |V|\}" />.
  </Text>
  <Text mb="md">
    <strong>Reconstruction</strong>: Text can be reconstructed by joining tokens with spaces: <InlineMath math="\text{Text} = \text{join}(\text{tokens}, \text{ })" />.
  </Text>
</Card>

<Title order={3} id="character-tokenization" mb="sm">
  Character-Level Tokenization
</Title>

<Text mb="md">
  Character tokenization breaks text into individual characters, offering a very small vocabulary but requiring longer sequences.
</Text>

<Card shadow="sm" p="md" mb="md" withBorder>
  <Title order={4} mb="sm">Formal Representation</Title>
  <Text mb="md">
    Given a character set <InlineMath math="\mathcal{C} = \{c_1, c_2, ..., c_{|\mathcal{C}|}\}" />:
  </Text>
  <BlockMath math="\text{Tokenize}_{char}: \mathcal{S} \rightarrow \mathcal{C}^*" />
  <Text mb="md">
    Where <InlineMath math="\mathcal{S}" /> is the space of all strings and <InlineMath math="\mathcal{C}^*" /> represents sequences of characters.
  </Text>
  <Text mb="md">
    <strong>Dimension</strong>: For a character set of size <InlineMath math="|\mathcal{C}|" /> (typically 26-128), each character is represented as an index in <InlineMath math="\{1, 2, ..., |\mathcal{C}|\}" />.
  </Text>
  <Text mb="md">
    <strong>Reconstruction</strong>: Text can be perfectly reconstructed by joining characters with no separator: <InlineMath math="\text{Text} = \text{join}(\text{char\_tokens}, \text{''})" />.
  </Text>
</Card>


<CodeBlock
  language="python"
  code={`
# Character tokenization with handling considerations
text = "Hello World!"

# Tokenization (text → character tokens)
char_tokens = list(text)
print("Character tokens:", char_tokens)
# Output: ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!']

# Case-insensitive variant
case_insensitive_chars = [c.lower() for c in char_tokens]
print("Case-insensitive chars:", case_insensitive_chars)
# Output: ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!']

# Create character vocabulary and map to indices
char_vocab = sorted(set(char_tokens))
char_to_idx = {char: idx for idx, char in enumerate(char_vocab, 1)}
print("Character vocabulary:", char_vocab)
# Output: [' ', '!', 'H', 'W', 'd', 'e', 'l', 'o', 'r']
print("Vocabulary size:", len(char_vocab))
# Output: Vocabulary size: 9

# Convert to numerical representation
numerical_chars = [char_to_idx[char] for char in char_tokens]
print("Numerical representation:", numerical_chars)
# Output: Numerical representation: [3, 6, 7, 7, 8, 1, 4, 8, 9, 7, 5, 2]

# Reconstruction (indices → text)
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
reconstructed_chars = [idx_to_char[idx] for idx in numerical_chars]
reconstructed_text = ''.join(reconstructed_chars)
print("Reconstructed text:", reconstructed_text)
# Output: Reconstructed text: Hello World!
  `}
/>


<Title order={3} id="subword-tokenization" mb="sm">
  Subword Tokenization
</Title>

<Text mb="md">
  Subword tokenization methods break words into meaningful subword units, balancing vocabulary size against semantic granularity.
  Three popular algorithms are Byte-Pair Encoding (BPE), WordPiece, and SentencePiece.
</Text>

<Title order={4} id="bpe" mb="sm">
  Byte-Pair Encoding (BPE)
</Title>

<Text mb="md">
  BPE iteratively merges the most frequent adjacent byte pairs or character pairs to form new subword tokens.
</Text>

<Card shadow="sm" p="md" mb="md" withBorder>
  <Title order={5} mb="sm">Formal Representation</Title>
  <Text mb="md">
    Given an initial character vocabulary <InlineMath math="\mathcal{C}" /> and a target vocabulary size <InlineMath math="k" />:
  </Text>
  <BlockMath math="\text{BPE}: \mathcal{C}^* \times \mathbb{N} \rightarrow \mathcal{V}^*" />
  <Text mb="md">
    Where <InlineMath math="\mathcal{V}" /> is the learned vocabulary with <InlineMath math="|\mathcal{V}| = |\mathcal{C}| + k" /> tokens after <InlineMath math="k" /> merge operations.
  </Text>
  <Text mb="md">
    <strong>Dimension</strong>: For a BPE vocabulary of size <InlineMath math="|\mathcal{V}|" /> (typically 30K-50K), each token is represented as an index in <InlineMath math="\{1, 2, ..., |\mathcal{V}|\}" />.
  </Text>
  <Text mb="md">
    <strong>Reconstruction</strong>: Text is reconstructed by concatenating the subword tokens: <InlineMath math="\text{Text} = \text{concatenate}(\text{subword\_tokens})" />.
  </Text>
</Card>


<Card shadow="sm" p="md" mb="md" withBorder>
  <Title order={5} mb="sm">Algorithm</Title>
  <Text component="div">
    <ol>
      <li>Start with a vocabulary of individual characters</li>
      <li>Count frequencies of adjacent character pairs in the corpus</li>
      <li>Merge the most frequent pair to create a new token</li>
      <li>Update frequencies with the new token</li>
      <li>Repeat steps 2-4 until reaching desired vocabulary size or frequency threshold</li>
    </ol>
  </Text>
</Card>

<Card shadow="sm" p="md" mb="lg" withBorder>
  <Title order={5} mb="sm">Step-by-Step Example</Title>
  <Text component="div">
    <p>Consider a tiny corpus: ["low", "lower", "lowest", "newer", "wider"]</p>
    <p><strong>Initial state:</strong></p>
    <p>Initial vocabulary: ['l', 'o', 'w', 'e', 'r', 's', 't', 'n', 'i', 'd'] (10 tokens)</p>
    <p>Initial segmentation: "l o w" (3 tokens), "l o w e r" (5 tokens), "l o w e s t" (6 tokens), "n e w e r" (5 tokens), "w i d e r" (5 tokens)</p>
    
    <p><strong>Iteration 1:</strong></p>
    <p>Frequency count: ('e', 'r') appears 3 times</p>
    <p>Merge 'e' + 'r' → 'er'</p>
    <p>Updated vocabulary: ['l', 'o', 'w', 'e', 'r', 's', 't', 'n', 'i', 'd', 'er'] (11 tokens)</p>
    <p>Updated segmentation: "l o w" (3 tokens), "l o w er" (4 tokens), "l o w e s t" (6 tokens), "n e w er" (4 tokens), "w i d er" (4 tokens)</p>
    
    <p><strong>Iteration 2:</strong></p>
    <p>Frequency count: ('l', 'o') appears 3 times</p>
    <p>Merge 'l' + 'o' → 'lo'</p>
    <p>Updated vocabulary: ['l', 'o', 'w', 'e', 'r', 's', 't', 'n', 'i', 'd', 'er', 'lo'] (12 tokens)</p>
    <p>Updated segmentation: "lo w" (2 tokens), "lo w er" (3 tokens), "lo w e s t" (5 tokens), "n e w er" (4 tokens), "w i d er" (4 tokens)</p>
    
    <p><strong>Iteration 3:</strong></p>
    <p>Frequency count: ('lo', 'w') appears 3 times</p>
    <p>Merge 'lo' + 'w' → 'low'</p>
    <p>Updated vocabulary: ['l', 'o', 'w', 'e', 'r', 's', 't', 'n', 'i', 'd', 'er', 'lo', 'low'] (13 tokens)</p>
    <p>Updated segmentation: "low" (1 token), "low er" (2 tokens), "low e s t" (4 tokens), "n e w er" (4 tokens), "w i d er" (4 tokens)</p>
    
    <p><strong>Final result after more iterations:</strong></p>
    <p>Vocabulary: ['l', 'o', 'w', 'e', 'r', 's', 't', 'n', 'i', 'd', 'er', 'lo', 'low', 'low er', 'est', ...] (15+ tokens)</p>
    <p>Final segmentation: "low" (1 token), "low er" (2 tokens), "low est" (2 tokens), "n e w er" (4 tokens), "w i d er" (4 tokens)</p>
  </Text>
</Card>

<CodeBlock
  language="python"
  code={`
# BPE tokenization example with reconstruction
from transformers import GPT2Tokenizer

# Load a pre-trained BPE tokenizer (GPT-2 uses BPE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Input text
text = "Natural language processing is powerful."

# Tokenization (text → BPE tokens)
tokens = tokenizer.tokenize(text)
print("BPE tokens:", tokens)
# Output: ['Natural', 'language', 'processing', 'is', 'powerful', '.']

# Convert to token IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)
# Output: [8241, 2061, 4915, 318, 5411, 13]

# Vocabulary size
print("Vocabulary size:", len(tokenizer))
# Output: Vocabulary size: 50257

# Reconstruction (token IDs → text)
reconstructed_text = tokenizer.decode(token_ids)
print("Reconstructed text:", reconstructed_text)
# Output: Reconstructed text: Natural language processing is powerful.
  `}
/>

<Title order={4} id="wordpiece" mb="sm">
  WordPiece
</Title>

<Text mb="md">
  WordPiece is similar to BPE but uses a likelihood-based criterion for merging tokens instead of frequency.
  It marks subword units that don't begin words with '##' to aid in reconstruction.
</Text>

<Card shadow="sm" p="md" mb="md" withBorder>
  <Title order={5} mb="sm">Formal Representation</Title>
  <Text mb="md">
    Similar to BPE, but with a different merge criterion:
  </Text>
  <BlockMath math="\text{WordPiece}: \mathcal{C}^* \times \mathbb{N} \rightarrow \mathcal{V}^*" />
  <Text mb="md">
    <strong>Dimension</strong>: For a WordPiece vocabulary of size <InlineMath math="|\mathcal{V}|" /> (typically ~30K), each token is represented as an index in <InlineMath math="\{1, 2, ..., |\mathcal{V}|\}" />.
  </Text>
  <Text mb="md">
    <strong>Reconstruction</strong>: Text is reconstructed by concatenating tokens and removing '##' markers: <InlineMath math="\text{Text} = \text{join}(\text{replace}(\text{tokens}, '\#\#', ''), '')" />.
  </Text>
</Card>


<Card shadow="sm" p="md" mb="lg" withBorder>
  <Title order={5} mb="sm">Decision Criterion</Title>
  <Text mb="md">
    WordPiece selects the merge that maximizes:
  </Text>
  <BlockMath math="\text{score}(x,y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}" />
  <Text>
    This formula represents a likelihood ratio that prefers merges where the combined token <InlineMath math="xy" /> appears more frequently than would be expected if <InlineMath math="x" /> and <InlineMath math="y" /> were independent.
  </Text>
</Card>

<CodeBlock
  language="python"
  code={`
# WordPiece tokenization example with reconstruction
from transformers import BertTokenizer

# Load a pre-trained WordPiece tokenizer (BERT uses WordPiece)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Input text
text = "Natural language processing uses wordpiece tokenization."

# Tokenization (text → WordPiece tokens)
tokens = tokenizer.tokenize(text)
print("WordPiece tokens:", tokens)
# Output: ['natural', 'language', 'processing', 'uses', 'word', '##piece', 'token', '##ization', '.']

# Notice how "wordpiece" is split into "word" and "##piece"
# The "##" prefix indicates this is a continuation of the previous token

# Convert to token IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)
# Output: [101, 2269, 2653, 6254, 2224, 2773, 12195, 2106, 10104, 1012, 102]
# (101 and 102 are special tokens for [CLS] and [SEP])

# Vocabulary size
print("Vocabulary size:", len(tokenizer))
# Output: Vocabulary size: 30522

# Reconstruction (token IDs → text)
reconstructed_text = tokenizer.decode(token_ids)
print("Reconstructed text:", reconstructed_text)
# Output: Reconstructed text: natural language processing uses wordpiece tokenization.
  `}
/>

{/* Section 3: Embeddings */}
<Title order={2} id="embeddings" mb="sm">
  Word and Token Embeddings
</Title>

<Text mb="md">
  Embeddings map tokens to dense vector representations in a continuous vector space, capturing semantic relationships.
  Unlike simple one-hot encoding, embeddings place similar words closer together in the vector space.
</Text>

<Title order={3} id="one-hot" mb="sm">
  One-Hot Encoding
</Title>

      <Text mb="md">
        The simplest numerical representation maps each token to a sparse vector with a single 1 and 0s elsewhere.
      </Text>

      <Card shadow="sm" p="md" mb="md" withBorder>
        <Title order={4} mb="sm">Mathematical Formulation</Title>
        <Text mb="md">
          For a vocabulary <InlineMath math="V" /> of size <InlineMath math="|V|" />, the one-hot encoding of token <InlineMath math="t_i" /> is a vector <InlineMath math="\mathbf{v}_i \in \{0,1\}^{|V|}" /> where:
        </Text>
        <BlockMath math="\mathbf{v}_i[j] = \begin{cases} 
          1 & \text{if } j = i \\
          0 & \text{otherwise}
          \end{cases}" />
      </Card>

      <CodeBlock
        language="python"
        code={`
# One-hot encoding example
vocabulary = ["natural", "language", "processing", "model"]
word = "language"
one_hot = [1 if token == word else 0 for token in vocabulary]
print(one_hot)
# Output: [0, 1, 0, 0]
        `}
      />

      <Text mb="md">
        One-hot encodings are sparse, high-dimensional, and fail to capture semantic relationships between words.
      </Text>

      <Title order={3} id="count-based" mb="sm">
        Count-Based Embeddings
      </Title>

      <Text mb="md">
        Count-based methods like TF-IDF and co-occurrence matrices capture word relationships based on their distributional statistics in a corpus.
      </Text>

      <Card shadow="sm" p="md" mb="md" withBorder>
        <Title order={4} mb="sm">TF-IDF</Title>
        <Text mb="md">
          Term Frequency-Inverse Document Frequency weighs terms based on their frequency in a document and rarity across documents:
        </Text>
        <BlockMath math="\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)" />
        <BlockMath math="\text{where } \text{IDF}(t, D) = \log\frac{|D|}{|\{d \in D: t \in d\}|}" />
        <Text>
          <InlineMath math="|D|" /> is the total number of documents and <InlineMath math="|\{d \in D: t \in d\}|" /> is the number of documents containing term <InlineMath math="t" />.
        </Text>
      </Card>

      <CodeBlock
        language="python"
        code={`
from sklearn.feature_extraction.text import TfidfVectorizer

# A clearer example corpus
corpus = [
    "apple apple orange",
    "apple banana banana",
    "cherry cherry cherry orange"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print("Matrix shape (documents × terms):", X.shape)
print("Sparse matrix representation:")
print(X)

print("Feature names (vocabulary):")
print(vectorizer.get_feature_names_out())

print("Full TF-IDF matrix:")
# Convert to dense array for better visualization
print(X.toarray())

print("Detailed coordinate interpretation:")
# Get coordinates and values from the sparse matrix
coords = X.nonzero()
for i, (doc_idx, term_idx) in enumerate(zip(coords[0], coords[1])):
    term = vectorizer.get_feature_names_out()[term_idx]
    value = X[doc_idx, term_idx]
    doc = corpus[doc_idx]
    print(f"Coord ({doc_idx}, {term_idx}): Document \"{doc}\" contains term \"{term}\" with TF-IDF score {value:.4f}")
# Output:
Matrix shape (documents × terms): (3, 4)

Sparse matrix representation:
<Compressed Sparse Row sparse matrix of dtype 'float64'
	with 6 stored elements and shape (3, 4)>
  Coords	Values
  (0, 0)	0.8944271909999159
  (0, 3)	0.4472135954999579
  (1, 0)	0.35543246785041743
  (1, 1)	0.9347019636214327
  (2, 3)	0.24573525337873806
  (2, 2)	0.9693369822960886

Feature names (vocabulary):
['apple' 'banana' 'cherry' 'orange']

Full TF-IDF matrix:
[[0.89442719 0.         0.         0.4472136 ]
 [0.35543247 0.93470196 0.         0.        ]
 [0.         0.         0.96933698 0.24573525]]

Detailed coordinate interpretation:
Coord (0, 0): Document "apple apple orange" contains term "apple" with TF-IDF score 0.8944
Coord (0, 3): Document "apple apple orange" contains term "orange" with TF-IDF score 0.4472
Coord (1, 0): Document "apple banana banana" contains term "apple" with TF-IDF score 0.3554
Coord (1, 1): Document "apple banana banana" contains term "banana" with TF-IDF score 0.9347
Coord (2, 3): Document "cherry cherry cherry orange" contains term "orange" with TF-IDF score 0.2457
Coord (2, 2): Document "cherry cherry cherry orange" contains term "cherry" with TF-IDF score 0.9693   `}
      />

      <LearnedEmbeddings />

      <Title order={3} id="contextual-embeddings" mb="sm">
        Contextual Embeddings
      </Title>

      <Text mb="md">
        Unlike static embeddings that assign the same vector to a word regardless of context, contextual embeddings generate
        different vectors based on the surrounding context. The word "bank" would have different embeddings in "river bank" vs. "bank account".
        We will see how to learn these embeddings in the next sections.
      </Text>


    </Container>
  );
};

export default TextNumericalRepresentation;