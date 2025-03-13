import React from "react";
import { Container, Title, Text, Card, Divider, List, Table } from "@mantine/core";
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";

const TextNumericalRepresentation = () => {
  return (
    <Container fluid>
      <Title order={1} id="numerical-representation" mb="md">
        Numerical Representation of Text
      </Title>
      
      <Text mb="lg">
        Converting text into numerical representations is a fundamental step in Natural Language Processing (NLP).
        This transformation allows machines to process and analyze human language using mathematical operations
        and algorithms. This page explores the challenges of text representation and the methods used to convert text
        into numerical formats, focusing on tokenization and embedding techniques.
      </Text>

      {/* Section 1: Challenges */}
      <Title order={2} id="challenges" mb="sm">
        Challenges of Numerical Text Representation
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

      <Title order={3} id="word-tokenization" mb="sm">
        Word-Level Tokenization
      </Title>

      <Text mb="md">
        Word tokenization splits text at word boundaries, typically using spaces and punctuation as delimiters.
      </Text>

      <CodeBlock
        language="python"
        code={`
# Simple word tokenization
text = "Natural language processing transforms text into numbers."
tokens = text.split()
print(tokens)
# Output: ['Natural', 'language', 'processing', 'transforms', 'text', 'into', 'numbers.']

# With punctuation handling
import re
tokens = re.findall(r'\w+|[^\w\s]', text)
print(tokens)
# Output: ['Natural', 'language', 'processing', 'transforms', 'text', 'into', 'numbers', '.']
        `}
      />

      <Card shadow="sm" p="md" mb="lg" withBorder>
        <Title order={4} mb="sm">Advantages and Limitations</Title>
        <Table>
          <thead>
            <tr>
              <th>Advantages</th>
              <th>Limitations</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Intuitive semantic units</td>
              <td>Large vocabulary size</td>
            </tr>
            <tr>
              <td>Preserves word boundaries</td>
              <td>Out-of-vocabulary (OOV) problem</td>
            </tr>
            <tr>
              <td>Simple implementation</td>
              <td>No morphological awareness</td>
            </tr>
          </tbody>
        </Table>
      </Card>

      <Title order={3} id="character-tokenization" mb="sm">
        Character-Level Tokenization
      </Title>

      <Text mb="md">
        Character tokenization breaks text into individual characters, offering a very small vocabulary but requiring longer sequences.
      </Text>

      <CodeBlock
        language="python"
        code={`
# Character tokenization
text = "Hello"
char_tokens = list(text)
print(char_tokens)
# Output: ['H', 'e', 'l', 'l', 'o']

# Converting to numerical representation
char_to_idx = {char: idx for idx, char in enumerate(sorted(set(char_tokens)))}
numerical_repr = [char_to_idx[char] for char in char_tokens]
print(numerical_repr)
# Output: [0, 2, 3, 3, 4] (exact output depends on character ordering)
        `}
      />

      <Card shadow="sm" p="md" mb="lg" withBorder>
        <Title order={4} mb="sm">Mathematical Formulation</Title>
        <Text mb="md">
          Character tokenization can be viewed as a mapping function:
        </Text>
        <BlockMath math="\mathcal{T}: \mathcal{S} \rightarrow \mathcal{C}^*" />
        <Text mb="md">
          where <InlineMath math="\mathcal{S}" /> is the space of all strings, <InlineMath math="\mathcal{C}" /> is the character vocabulary, 
          and <InlineMath math="\mathcal{C}^*" /> represents sequences of characters.
        </Text>
      </Card>

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
        <Title order={5} mb="sm">Simple Example</Title>
        <Text component="div">
          <p>Consider a tiny corpus: ["low", "lower", "lowest", "newer", "wider"]</p>
          <p>Initial vocabulary: ['l', 'o', 'w', 'e', 'r', 's', 't', 'n', 'i', 'd']</p>
          <p>Initial segmentation: "l o w", "l o w e r", "l o w e s t", "n e w e r", "w i d e r"</p>
          <p>Most frequent pair: ('e', 'r') appearing 3 times → merge into 'er'</p>
          <p>Updated segmentation: "l o w", "l o w er", "l o w e s t", "n e w er", "w i d er"</p>
          <p>Next most frequent pair: ('l', 'o') appearing 3 times → merge into 'lo'</p>
          <p>Updated segmentation: "lo w", "lo w er", "lo w e s t", "n e w er", "w i d er"</p>
        </Text>
      </Card>

      <CodeBlock
        language="python"
        code={`
# Using BPE tokenizer from Hugging Face
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "Natural language processing is powerful."
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['Natural', 'language', 'processing', 'is', 'powerful', '.']

# Get the token IDs
token_ids = tokenizer.encode(text)
print(token_ids)
# Output: [8241, 2061, 4915, 318, 5411, 13]
        `}
      />

      <Title order={4} id="wordpiece" mb="sm">
        WordPiece
      </Title>

      <Text mb="md">
        WordPiece is similar to BPE but uses a different criterion for merging tokens. Instead of frequency,
        it uses a likelihood criterion that maximizes the language model probability.
      </Text>

      <Card shadow="sm" p="md" mb="lg" withBorder>
        <Title order={5} mb="sm">Mathematical Decision Criterion</Title>
        <Text mb="md">
          WordPiece selects the merge that maximizes:
        </Text>
        <BlockMath math="\text{score}(x,y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}" />
        <Text>
          Where <InlineMath math="\text{freq}(x)" /> is the frequency of token <InlineMath math="x" />, and
          <InlineMath math="\text{freq}(xy)" /> is the frequency of tokens <InlineMath math="x" /> and <InlineMath math="y" /> appearing consecutively.
        </Text>
      </Card>

      <CodeBlock
        language="python"
        code={`
# Using BERT's WordPiece tokenizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "Natural language processing uses wordpiece tokenization."
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['natural', 'language', 'processing', 'uses', 'word', '##piece', 'token', '##ization', '.']
        `}
      />

      <Title order={4} id="sentencepiece" mb="sm">
        SentencePiece
      </Title>

      <Text mb="md">
        SentencePiece treats the text as a sequence of Unicode characters and doesn't require pre-tokenization, making
        it language-agnostic and suitable for languages without clear word boundaries.
      </Text>

      <CodeBlock
        language="python"
        code={`
# Using SentencePiece with a T5 model
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
text = "Natural language processing with SentencePiece."
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['▁Natural', '▁language', '▁processing', '▁with', '▁Sentence', 'Piece', '.']
        `}
      />

      <Text mb="md">
        The "▁" symbol (underscore) represents a space, indicating word boundaries in SentencePiece.
      </Text>

      <Card shadow="sm" p="md" mb="lg" withBorder>
        <Title order={3} mb="sm">Tokenization Comparison</Title>
        <Table>
          <thead>
            <tr>
              <th>Tokenizer</th>
              <th>Vocabulary Size</th>
              <th>OOV Handling</th>
              <th>Language Support</th>
              <th>Common Use Cases</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Word-level</td>
              <td>Very large (>100K)</td>
              <td>Poor (UNK tokens)</td>
              <td>Language-specific</td>
              <td>Traditional NLP</td>
            </tr>
            <tr>
              <td>Character-level</td>
              <td>Very small (26-128)</td>
              <td>Excellent</td>
              <td>Universal</td>
              <td>Shallow networks, morphologically rich languages</td>
            </tr>
            <tr>
              <td>BPE</td>
              <td>Medium (30K-50K)</td>
              <td>Good</td>
              <td>Multilingual</td>
              <td>GPT models, machine translation</td>
            </tr>
            <tr>
              <td>WordPiece</td>
              <td>Medium (30K)</td>
              <td>Good</td>
              <td>Good for Western</td>
              <td>BERT models</td>
            </tr>
            <tr>
              <td>SentencePiece</td>
              <td>Configurable</td>
              <td>Excellent</td>
              <td>Universal</td>
              <td>T5, XLM-R, multilingual models</td>
            </tr>
          </tbody>
        </Table>
      </Card>

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
# TF-IDF with scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Natural language processing is fascinating.",
    "Processing text requires specialized techniques.",
    "Language models help with text analysis."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.shape)  # Number of documents × vocabulary size
# Output: (3, 11)

# Get the feature names (tokens)
print(vectorizer.get_feature_names_out())
# Output: ['analysis' 'fascinating' 'help' 'is' 'language' 'models' 'natural' 'processing' 'requires' 'specialized' 'text' 'techniques' 'with']
        `}
      />

      <Title order={3} id="learned-embeddings" mb="sm">
        Learned Embeddings
      </Title>

      <Text mb="md">
        Learned embeddings create dense vector representations by training on large text corpora, capturing semantic and
        syntactic relationships between words.
      </Text>

      <Title order={4} id="word2vec" mb="sm">
        Word2Vec
      </Title>

      <Text mb="md">
        Word2Vec learns word embeddings by predicting either a word from its context (CBOW) or context from a word (Skip-gram).
      </Text>

      <Card shadow="sm" p="md" mb="md" withBorder>
        <Title order={5} mb="sm">Skip-gram Model</Title>
        <Text mb="md">
          The skip-gram model maximizes the probability of context words given a target word:
        </Text>
        <BlockMath math="\mathcal{L} = \sum_{w \in \text{corpus}} \sum_{c \in \text{context}(w)} \log P(c|w)" />
        <Text mb="md">
          Where <InlineMath math="P(c|w)" /> is modeled using the softmax function:
        </Text>
        <BlockMath math="P(c|w) = \frac{\exp(\mathbf{v}_w^T \mathbf{v}_c')}{\sum_{c' \in V} \exp(\mathbf{v}_w^T \mathbf{v}_{c'}')}"/>
        <Text>
          <InlineMath math="\mathbf{v}_w" /> is the embedding of the target word and <InlineMath math="\mathbf{v}_c'" /> is the context word embedding.
        </Text>
      </Card>

      <CodeBlock
        language="python"
        code={`
# Using pre-trained Word2Vec embeddings
import gensim.downloader as api

# Load pre-trained Word2Vec embeddings
word2vec_model = api.load("word2vec-google-news-300")

# Get vector for a word
vector = word2vec_model["language"]  # 300-dimensional vector

# Find similar words
similar_words = word2vec_model.most_similar("language", topn=3)
print(similar_words)
# Output: [('languages', 0.7182), ('linguistic', 0.6953), ('grammar', 0.6522)]
        `}
      />

      <Title order={4} id="glove" mb="sm">
        GloVe (Global Vectors)
      </Title>

      <Text mb="md">
        GloVe combines count-based and prediction-based methods by training on global word-word co-occurrence statistics.
      </Text>

      <Card shadow="sm" p="md" mb="md" withBorder>
        <Title order={5} mb="sm">Mathematical Objective</Title>
        <Text mb="md">
          GloVe minimizes the following cost function:
        </Text>
        <BlockMath math="J = \sum_{i,j=1}^{|V|} f(X_{ij})(\mathbf{w}_i^T\tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2" />
        <Text>
          Where <InlineMath math="X_{ij}" /> is the co-occurrence count of words <InlineMath math="i" /> and <InlineMath math="j" />,
          <InlineMath math="f" /> is a weighting function, and <InlineMath math="\mathbf{w}_i" /> and <InlineMath math="\tilde{\mathbf{w}}_j" /> are word vectors.
        </Text>
      </Card>

      <Title order={4} id="fasttext" mb="sm">
        FastText
      </Title>

      <Text mb="md">
        FastText extends Word2Vec by representing each word as a bag of character n-grams, allowing it to capture morphological information and handle out-of-vocabulary words.
      </Text>

      <Card shadow="sm" p="md" mb="md" withBorder>
        <Title order={5} mb="sm">Word Representation</Title>
        <Text mb="md">
          In FastText, a word <InlineMath math="w" /> is represented as:
        </Text>
        <BlockMath math="\mathbf{v}_w = \frac{1}{|G_w|} \sum_{g \in G_w} \mathbf{z}_g" />
        <Text>
          Where <InlineMath math="G_w" /> is the set of n-grams in <InlineMath math="w" /> and <InlineMath math="\mathbf{z}_g" /> is the vector representation of n-gram <InlineMath math="g" />.
        </Text>
      </Card>

      <CodeBlock
        language="python"
        code={`
# Using FastText for OOV words
from gensim.models import FastText

# Sample corpus
sentences = [["natural", "language", "processing"], 
             ["deep", "learning", "models"]]

# Train a FastText model
model = FastText(sentences, vector_size=10, window=3, min_count=1, epochs=10)

# Even for unseen words, FastText can generate embeddings
vector = model.wv["unseen_word"]
        `}
      />

      <Title order={3} id="contextual-embeddings" mb="sm">
        Contextual Embeddings
      </Title>

      <Text mb="md">
        Unlike static embeddings that assign the same vector to a word regardless of context, contextual embeddings generate
        different vectors based on the surrounding context.
      </Text>

      <Card shadow="sm" p="md" mb="md" withBorder>
        <Title order={4} mb="sm">BERT Embeddings</Title>
        <Text>
          BERT (Bidirectional Encoder Representations from Transformers) generates embeddings that capture different word senses
          based on bidirectional context. The word "bank" would have different embeddings in "river bank" vs. "bank account".
        </Text>
      </Card>

      <CodeBlock
        language="python"
        code={`
# Using BERT contextual embeddings
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
text = "The bank by the river is beautiful."
inputs = tokenizer(text, return_tensors="pt")

# Get contextual embeddings
with torch.no_grad():
    outputs = model(**inputs)
    
# Get embedding for 'bank' (token at position 2)
bank_embedding = outputs.last_hidden_state[0, 2, :]

# Different context
text2 = "I need to deposit money at the bank."
inputs2 = tokenizer(text2, return_tensors="pt")

with torch.no_grad():
    outputs2 = model(**inputs2)
    
# Get embedding for 'bank' (token at position 8)
bank_embedding2 = outputs2.last_hidden_state[0, 8, :]

# The two embeddings for 'bank' will differ based on context
        `}
      />

    </Container>
  );
};

export default TextNumericalRepresentation;