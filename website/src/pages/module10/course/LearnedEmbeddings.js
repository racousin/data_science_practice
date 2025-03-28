import React, { useState } from "react";
import { Container, Title, Text, Card, Divider, Accordion, Table, Image } from "@mantine/core";
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";

const LearnedEmbeddings = () => {

  return (
    <>
      <Title order={3} id="learned-embeddings" mb="sm">
        Learned Embeddings
      </Title>

      <Text mb="md">
        Learned embeddings create dense vector representations by training on large text corpora, capturing semantic and 
        syntactic relationships between words. Unlike one-hot encodings or simple count-based methods, these models learn 
        to position words in a continuous vector space where semantic relationships are preserved as geometric properties.
      </Text>


      <Card shadow="sm" p="md" mb="md" withBorder>
        <Title order={4} mb="sm">Formal Definition</Title>
        <Text mb="md">
          A learned embedding model creates a function <InlineMath math="E: \mathcal{V} \rightarrow \mathbb{R}^d" /> that maps words from vocabulary <InlineMath math="\mathcal{V}" /> to vectors in <InlineMath math="d" />-dimensional space.
        </Text>
        <Text mb="md">
          The embedding function is learned by optimizing an objective function <InlineMath math="\mathcal{L}" /> that encodes the desired properties:
        </Text>
        <BlockMath math="\mathbf{E} = \arg\min_{\mathbf{E}} \mathcal{L}(\mathbf{E}, \mathcal{D})" />
        <Text mb="md">
          Where <InlineMath math="\mathcal{D}" /> is the training corpus and <InlineMath math="\mathcal{L}" /> varies by embedding approach.
        </Text>
        <Text mb="md">
          The learned embedding matrix <InlineMath math="\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}" /> contains embeddings for all vocabulary words:
        </Text>
        <BlockMath math="\mathbf{E} = \begin{bmatrix} 
          \vec{e}_1 \\
          \vec{e}_2 \\
          \vdots \\
          \vec{e}_{|\mathcal{V}|}
          \end{bmatrix}" />
        <Text mb="md">
          Semantic similarity between words <InlineMath math="w_i" /> and <InlineMath math="w_j" /> is measured using cosine similarity:
        </Text>
        <BlockMath math="\text{similarity}(w_i, w_j) = \frac{\vec{e}_i \cdot \vec{e}_j}{||\vec{e}_i|| \cdot ||\vec{e}_j||}" />
      </Card>

<Card shadow="sm" p="md" mb="lg" withBorder>
  <Title order={4} mb="sm">Example</Title>
  <Text mb="md">
    Consider a simplified 2-dimensional embedding space where words are positioned based on their semantic properties:
  </Text>
  <Text mb="md">
    <ul>
      <li>Dimension 1 (x-axis) might capture gender attributes</li>
      <li>Dimension 2 (y-axis) might capture size or magnitude attributes</li>
    </ul>
  </Text>
  <Text mb="md">
    In this space, similar words cluster together, and interesting semantic relationships emerge as directions:
  </Text>
  <Text mb="md">
    <ul>
      <li>The vector from "king" to "queen" points in a similar direction as "man" to "woman" (gender dimension)</li>
      <li>The vector from "cat" to "kitten" points in a similar direction as "dog" to "puppy" (age dimension)</li>
      <li>Words like "big", "large", and "huge" cluster together, as do their antonyms "small", "tiny", and "little"</li>
    </ul>
  </Text>
  <Text mb="md">
    These semantic relationships can be leveraged for analogical reasoning: if <InlineMath math="\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}" />, then <InlineMath math="\vec{v}_{\text{king}} - \vec{v}_{\text{queen}} \approx \vec{v}_{\text{man}} - \vec{v}_{\text{woman}}" />.
  </Text>
</Card>
<Image
              src="/assets/module10/word_embeddings.png"
              alt="Word Embedding Example"
              caption="Word Embedding Example"
            />

      <Accordion 
        multiple 
        variant="separated"
        mt="md"
      >
        <Accordion.Item value="word2vec">
          <Accordion.Control>
            <Title order={4}>Word2Vec</Title>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              Word2Vec, introduced by Mikolov et al. (2013), learns word embeddings through shallow neural networks optimized 
              to predict words from context or vice versa.
            </Text>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Model Architecture</Title>
              <Text mb="md">
                Word2Vec has two distinct architectures:
              </Text>
              <Text mb="md" component="div">
                <ol>
                  <li><strong>CBOW (Continuous Bag of Words)</strong>: Predicts a target word given its context words</li>
                  <li><strong>Skip-gram</strong>: Predicts context words given a target word</li>
                </ol>
              </Text>
            </Card>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Skip-gram Model</Title>
              <Text mb="md">
                <strong>Input</strong>: 
              </Text>
              <Text component="div" mb="md">
                <ul>
                  <li>One-hot encoded target word vector <InlineMath math="\mathbf{x} \in \{0,1\}^{|V|}" /></li>
                  <li>Context window size <InlineMath math="c" /> (typically 5-10)</li>
                </ul>
              </Text>

              <Text mb="md">
                <strong>Parameters</strong>:
              </Text>
              <Text component="div" mb="md">
                <ul>
                  <li>Input embedding matrix <InlineMath math="\mathbf{W} \in \mathbb{R}^{|V| \times d}" /> (target word embeddings)</li>
                  <li>Output embedding matrix <InlineMath math="\mathbf{W'} \in \mathbb{R}^{d \times |V|}" /> (context word embeddings)</li>
                  <li>Dimensionality <InlineMath math="d" /> of the embedding space (typically 100-300)</li>
                </ul>
              </Text>

              <Text mb="md">
                <strong>Forward Pass</strong>:
              </Text>
              <Text component="div" mb="md">
                <ol>
                  <li>Embedding lookup: <InlineMath math="\mathbf{h} = \mathbf{W}^T\mathbf{x}" /> (extract the embedding for the target word)</li>
                  <li>For each position in the context window, compute probability distribution over vocabulary:
                    <InlineMath math="\mathbf{y} = \text{softmax}(\mathbf{W'h})" /></li>
                </ol>
              </Text>
              
              <Text mb="md">
                <strong>Training Objective</strong>:
              </Text>
              <Text mb="md">
                The Skip-gram model maximizes the log probability of context words given a target word:
              </Text>
              <BlockMath math="\mathcal{L} = \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j}|w_t)" />
              <Text mb="md">
                Where:
              </Text>
              <Text component="div" mb="md">
                <ul>
                  <li><InlineMath math="T" /> is the total number of words in the corpus</li>
                  <li><InlineMath math="c" /> is the context window size</li>
                  <li><InlineMath math="P(w_{t+j}|w_t)" /> is modeled using the softmax function:</li>
                </ul>
              </Text>
              <BlockMath math="P(w_j|w_i) = \frac{\exp(\mathbf{v}_{w_i}^T \mathbf{v}'_{w_j})}{\sum_{w \in V} \exp(\mathbf{v}_{w_i}^T \mathbf{v}'_w)}" />
              <Text mb="md">
                Where <InlineMath math="\mathbf{v}_{w_i}" /> is the input embedding of word <InlineMath math="w_i" /> and <InlineMath math="\mathbf{v}'_{w_j}" /> is the output embedding of word <InlineMath math="w_j" />.
              </Text>

              <Text mb="md">
                <strong>Optimization Improvements</strong>:
              </Text>
              <Text mb="md">
                Computing the full softmax is computationally expensive for large vocabularies. Two main optimization techniques are:
              </Text>
              <Text mb="md">
                1. <strong>Negative Sampling</strong>: Instead of computing the full softmax, the model is trained to distinguish the correct context word from randomly sampled "negative" words:
              </Text>
              <BlockMath math="\log \sigma(\mathbf{v}_{w_i}^T \mathbf{v}'_{w_j}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} [\log \sigma(-\mathbf{v}_{w_i}^T \mathbf{v}'_{w_k})]" />
              <Text mb="md">
                Where <InlineMath math="K" /> is the number of negative samples (typically 5-20) and <InlineMath math="P_n(w)" /> is a noise distribution (typically <InlineMath math="P_n(w) \propto f(w)^{3/4}" /> where <InlineMath math="f(w)" /> is the word frequency).
              </Text>
              <Text mb="md">
                2. <strong>Hierarchical Softmax</strong>: Uses a binary tree structure to represent the vocabulary, reducing the complexity from <InlineMath math="O(|V|)" /> to <InlineMath math="O(\log |V|)" />.
              </Text>
            </Card>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">CBOW Model</Title>
              <Text mb="md">
                <strong>Input</strong>: 
              </Text>
              <Text mb="md">
                Context words represented as one-hot vectors <InlineMath math="\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_{2c} \in \{0,1\}^{|V|}" />
              </Text>

              <Text mb="md">
                <strong>Parameters</strong>:
              </Text>
              <Text mb="md">
                Same embedding matrices as Skip-gram
              </Text>

              <Text mb="md">
                <strong>Forward Pass</strong>:
              </Text>
              <Text component="div" mb="md">
                <ol>
                  <li>Average context word embeddings: <InlineMath math="\mathbf{h} = \frac{1}{2c}\sum_{i=1}^{2c} \mathbf{W}^T\mathbf{x}_i" /></li>
                  <li>Predict target word: <InlineMath math="\mathbf{y} = \text{softmax}(\mathbf{W'h})" /></li>
                </ol>
              </Text>
              
              <Text mb="md">
                <strong>Training Objective</strong>:
              </Text>
              <Text mb="md">
                The CBOW model maximizes the log probability of the target word given the context:
              </Text>
              <BlockMath math="\mathcal{L} = \sum_{t=1}^{T} \log P(w_t|w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})" />
            </Card>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Output Representation</Title>
              <Text mb="md">
                After training, the input matrix <InlineMath math="\mathbf{W}" /> is typically used as the final word embedding matrix. Each row <InlineMath math="\mathbf{W}_i" /> represents the embedding of word <InlineMath math="i" /> in the vocabulary.
              </Text>
              <Text mb="md">
                <strong>Prediction</strong>:
                Once trained, the model is discarded, and embeddings are used directly for downstream tasks. Word similarity is computed using cosine similarity:
              </Text>
              <BlockMath math="\text{similarity}(w_i, w_j) = \frac{\mathbf{v}_{w_i} \cdot \mathbf{v}_{w_j}}{||\mathbf{v}_{w_i}|| \cdot ||\mathbf{v}_{w_j}||}" />
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
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="glove">
          <Accordion.Control>
            <Title order={4}>GloVe (Global Vectors)</Title>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              GloVe, introduced by Pennington et al. (2014), combines count-based and prediction-based approaches by training on global word-word co-occurrence statistics.
            </Text>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Model Architecture</Title>
              <Text mb="md">
                <strong>Preprocessing</strong>:
              </Text>
              <Text component="div" mb="md">
                <ol>
                  <li>Build a co-occurrence matrix <InlineMath math="X" /> where <InlineMath math="X_{ij}" /> represents how often word <InlineMath math="i" /> appears in the context of word <InlineMath math="j" /></li>
                  <li>Define a context window size (typically 5-10 words)</li>
                  <li>Optionally apply distance weighting: words closer to the target word contribute more to the co-occurrence count</li>
                </ol>
              </Text>

              <Text mb="md">
                <strong>Parameters</strong>:
              </Text>
              <Text component="div" mb="md">
                <ul>
                  <li>Word vectors <InlineMath math="\mathbf{w}_i \in \mathbb{R}^d" /> for each word <InlineMath math="i" /></li>
                  <li>Context vectors <InlineMath math="\tilde{\mathbf{w}}_j \in \mathbb{R}^d" /> for each word <InlineMath math="j" /></li>
                  <li>Word biases <InlineMath math="b_i" /> and context biases <InlineMath math="\tilde{b}_j" /></li>
                  <li>Dimensionality <InlineMath math="d" /> of the embedding space (typically 100-300)</li>
                </ul>
              </Text>
            </Card>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Training Objective</Title>
              <Text mb="md">
                GloVe minimizes the following cost function:
              </Text>
              <BlockMath math="J = \sum_{i,j=1}^{|V|} f(X_{ij})(\mathbf{w}_i^T\tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2" />
              <Text mb="md">
                Where:
              </Text>
              <Text component="div" mb="md">
                <ul>
                  <li><InlineMath math="X_{ij}" /> is the co-occurrence count of words <InlineMath math="i" /> and <InlineMath math="j" /></li>
                  <li><InlineMath math="f" /> is a weighting function that prevents rare co-occurrences from being overweighted:</li>
                </ul>
              </Text>
              <BlockMath math="f(x) = \begin{cases}
                (x/x_{\max})^{\alpha} & \text{if } x < x_{\max} \\
                1 & \text{otherwise}
                \end{cases}" />
              <Text mb="md">
                Typically, <InlineMath math="\alpha = 0.75" /> and <InlineMath math="x_{\max} = 100" />.
              </Text>
            </Card>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Key Insight</Title>
              <Text mb="md">
                The core insight of GloVe is that ratios of co-occurrence probabilities can encode meaning. For example, in a corpus about ice and steam, the ratio <InlineMath math="P_{ik}/P_{jk}" /> will be large if word <InlineMath math="k" /> is related to ice (<InlineMath math="i" />) but not steam (<InlineMath math="j" />).
              </Text>
            </Card>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Training and Output</Title>
              <Text mb="md">
                <strong>Training</strong>:
              </Text>
              <Text component="div" mb="md">
                <ul>
                  <li>Unlike Word2Vec, GloVe doesn't use gradient descent on a neural network</li>
                  <li>Instead, it uses stochastic gradient descent directly on the cost function</li>
                  <li>Typically runs for 50-100 epochs with decreasing learning rate</li>
                </ul>
              </Text>

              <Text mb="md">
                <strong>Output Representation</strong>:
              </Text>
              <Text mb="md">
                After training, the final word embeddings are typically defined as:
              </Text>
              <BlockMath math="\mathbf{v}_{final} = \mathbf{w} + \tilde{\mathbf{w}}" />
            </Card>

            <CodeBlock
              language="python"
              code={`
# Using pre-trained GloVe embeddings
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Convert GloVe format to Word2Vec format
glove_input_file = "glove.6B.100d.txt"
word2vec_output_file = "glove.6B.100d.word2vec.txt"
glove2word2vec(glove_input_file, word2vec_output_file)

# Load the converted embeddings
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file)

# Get vector for a word
vector = glove_model["language"]  # 100-dimensional vector

# Find similar words
similar_words = glove_model.most_similar("language", topn=3)
print(similar_words)
              `}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="fasttext">
          <Accordion.Control>
            <Title order={4}>FastText</Title>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              FastText, developed by Facebook Research (Bojanowski et al., 2017), extends Word2Vec by representing words as bags of character n-grams, allowing it to capture morphological information and handle out-of-vocabulary words.
            </Text>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Input Representation</Title>
              <Text mb="md">
                Each word is represented as a bag of character n-grams plus the word itself.
              </Text>
              <Text mb="md">
                For example, with n-grams of length 3-6, the word "where" would be represented as:
              </Text>
              <Text component="div" mb="md">
                <ul>
                  <li>The word itself: "where"</li>
                  <li>Character 3-grams: "&lt;wh", "whe", "her", "ere", "re&gt;"</li>
                  <li>Character 4-grams: "&lt;whe", "wher", "here", "ere&gt;"</li>
                  <li>Character 5-grams: "&lt;wher", "where", "here&gt;"</li>
                  <li>Character 6-grams: "&lt;where", "where&gt;"</li>
                </ul>
              </Text>
              <Text mb="md">
                Where "&lt;" and "&gt;" represent word boundary markers.
              </Text>
            </Card>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Model Architecture</Title>
              <Text mb="md">
                <strong>Parameters</strong>:
              </Text>
              <Text component="div" mb="md">
                <ul>
                  <li>Embedding matrix <InlineMath math="\mathbf{Z} \in \mathbb{R}^{N \times d}" /> for all n-grams in the corpus</li>
                  <li><InlineMath math="N" /> is the total number of unique n-grams plus words</li>
                  <li><InlineMath math="d" /> is the embedding dimension (typically 100-300)</li>
                </ul>
              </Text>

              <Text mb="md">
                <strong>Word Representation</strong>:
              </Text>
              <Text mb="md">
                A word <InlineMath math="w" /> is represented as the sum of its n-gram embeddings:
              </Text>
              <BlockMath math="\mathbf{v}_w = \sum_{g \in G_w} \mathbf{z}_g" />
              <Text mb="md">
                Where <InlineMath math="G_w" /> is the set of n-grams in <InlineMath math="w" /> (including the word itself) and <InlineMath math="\mathbf{z}_g" /> is the vector for n-gram <InlineMath math="g" />.
              </Text>
            </Card>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Training Objective</Title>
              <Text mb="md">
                FastText can use either the CBOW or Skip-gram architecture from Word2Vec, with the same loss functions but with word representations as sums of n-gram vectors:
              </Text>
              <Text mb="md">
                For Skip-gram:
              </Text>
              <BlockMath math="\mathcal{L} = \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j}|w_t)" />
              <Text mb="md">
                Where:
              </Text>
              <BlockMath math="P(w_j|w_i) = \frac{\exp(\mathbf{v}_{w_i}^T \mathbf{v}'_{w_j})}{\sum_{w \in V} \exp(\mathbf{v}_{w_i}^T \mathbf{v}'_w)}" />
              <Text mb="md">
                But now <InlineMath math="\mathbf{v}_{w_i} = \sum_{g \in G_{w_i}} \mathbf{z}_g" /> is the sum of n-gram vectors.
              </Text>
              <Text mb="md">
                <strong>Optimization</strong>:
              </Text>
              <Text mb="md">
                Like Word2Vec, FastText uses negative sampling to efficiently approximate the softmax.
              </Text>
            </Card>

            <Card shadow="sm" p="md" mb="md" withBorder>
              <Title order={5} mb="sm">Key Advantages</Title>
              <Text component="div" mb="md">
                <ol>
                  <li><strong>Handling Out-of-Vocabulary Words</strong>: Even for unseen words, FastText can generate embeddings by summing the embeddings of their character n-grams.</li>
                  <li><strong>Morphological Awareness</strong>: Words with similar morphological structures (e.g., "play", "plays", "playing") will share many n-grams and thus have similar embeddings.</li>
                  <li><strong>Rare Word Representation</strong>: Rare words benefit from sharing subcomponents with more common words.</li>
                </ol>
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
model = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=10)

# Vector for word in training data
vector = model.wv["language"]  # 100-dimensional vector

# Even for unseen words, FastText can generate embeddings
unseen_vector = model.wv["multilingualism"]

# Find similar words to the unseen word
similar_to_unseen = model.wv.most_similar("multilingualism", topn=3)
print(similar_to_unseen)
              `}
            />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

    </>
  );
};

export default LearnedEmbeddings;