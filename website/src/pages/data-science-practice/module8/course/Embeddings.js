import React from "react";
import { Text, Title, List, Table, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";

const Embeddings = () => {
  return (
    <>
      <div data-slide>
        <Title order={2} mb="md">
          Embeddings
        </Title>
        <Text size="lg" mb="md">
          Dense vector representations that capture semantic meaning
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          What Are Embeddings?
        </Title>
        <Text mb="md">
          Embeddings are dense vector representations of words, tokens, or other entities in a continuous vector space. Unlike sparse encodings like one-hot vectors, embeddings capture semantic and syntactic relationships in their geometric properties.
        </Text>
        <Text mb="md">
          Key characteristics of embeddings:
        </Text>
        <List spacing="sm">
          <List.Item>Dense representations (100-300 dimensions instead of vocabulary size)</List.Item>
          <List.Item>Learned from data rather than manually designed</List.Item>
          <List.Item>Semantic similarity correlates with vector distance</List.Item>
          <List.Item>Support arithmetic operations that capture analogies</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          From One-Hot to Dense Embeddings
        </Title>
        <Text mb="md">
          Consider a vocabulary of 10,000 words:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item><strong>One-Hot Encoding</strong>: 10,000-dimensional vector with a single 1 and 9,999 zeros. No notion of similarity between words.</List.Item>
          <List.Item><strong>Dense Embedding</strong>: 300-dimensional vector with real values. Similar words have similar vectors.</List.Item>
        </List>
        <Text>
          This dimensionality reduction from 10,000 to 300 makes embeddings computationally efficient while capturing much richer semantic information.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Dimensionality Reduction Visualization
        </Title>
        <Image
          src="/assets/data-science-practice/module8/dimensionality-reduction.png"
          alt="One-Hot vs Dense Embeddings"
          caption="Visualization comparing sparse one-hot encoding to dense embeddings"
        />
        <Text size="sm" mt="md">
          Dense embeddings dramatically reduce dimensionality while capturing semantic relationships.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Learned Embeddings Introduction
        </Title>
        <Text mb="md">
          Learned embeddings create dense vector representations by training on large text corpora, capturing semantic and syntactic relationships between words. Unlike one-hot encodings or simple count-based methods, these models learn to position words in a continuous vector space where semantic relationships are preserved as geometric properties.
        </Text>
        <Text>
          The three major approaches to learned embeddings are:
        </Text>
        <List spacing="sm" mt="sm">
          <List.Item><strong>Word2Vec</strong>: Predicts context from words or words from context</List.Item>
          <List.Item><strong>GloVe</strong>: Factorizes global co-occurrence statistics</List.Item>
          <List.Item><strong>FastText</strong>: Extends Word2Vec with character n-grams</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Word Embeddings Visualization
        </Title>
        <Image
          src="/assets/data-science-practice/module8/word-embeddings-visualization.png"
          alt="Word Embeddings Visualization"
          caption="Visualization of word embeddings in reduced dimensional space"
        />
        <Text size="sm" mt="md">
          Word embeddings map discrete words to continuous vector spaces where semantic relationships are preserved geometrically.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Formal Definition
        </Title>
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
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Embedding Matrix Structure
        </Title>
        <Text mb="md">
          The learned embedding matrix <InlineMath math="\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}" /> contains embeddings for all vocabulary words:
        </Text>
        <BlockMath math="\mathbf{E} = \begin{bmatrix}
          \vec{e}_1 \\
          \vec{e}_2 \\
          \vdots \\
          \vec{e}_{|\mathcal{V}|}
          \end{bmatrix}" />
        <Text>
          Each row <InlineMath math="\vec{e}_i \in \mathbb{R}^d" /> represents the embedding for word <InlineMath math="i" /> in the vocabulary.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Embedding Space Properties
        </Title>
        <Text mb="md">
          Semantic similarity between words <InlineMath math="w_i" /> and <InlineMath math="w_j" /> is measured using cosine similarity:
        </Text>
        <BlockMath math="\text{similarity}(w_i, w_j) = \frac{\vec{e}_i \cdot \vec{e}_j}{||\vec{e}_i|| \cdot ||\vec{e}_i||}" />
        <Text mt="md">
          Semantic relationships emerge as directions in the embedding space:
        </Text>
        <List spacing="sm" mt="sm">
          <List.Item>The vector from "king" to "queen" points in a similar direction as "man" to "woman" (gender)</List.Item>
          <List.Item>The vector from "cat" to "kitten" is similar to "dog" to "puppy" (age)</List.Item>
          <List.Item>Words like "big", "large", "huge" cluster together</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Semantic Clustering
        </Title>
        <Image
          src="/assets/data-science-practice/module8/semantic-clustering.png"
          alt="Semantic Clustering in Embeddings"
          caption="Visualization of semantic clustering in embedding space"
        />
        <Text size="sm" mt="md">
          Similar words cluster together in the embedding space, with distances reflecting semantic relatedness.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Analogical Reasoning
        </Title>
        <Text mb="md">
          These semantic relationships can be leveraged for analogical reasoning:
        </Text>
        <BlockMath math="\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}" />
        <Text mb="md">
          This property shows that:
        </Text>
        <BlockMath math="\vec{v}_{\text{king}} - \vec{v}_{\text{queen}} \approx \vec{v}_{\text{man}} - \vec{v}_{\text{woman}}" />
        <Text>
          The difference vectors capture the concept of "gender" in this embedding space.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Vector Arithmetic Visualization
        </Title>
        <Image
          src="/assets/data-science-practice/module8/vector-arithmetic.png"
          alt="Vector Arithmetic in Embeddings"
          caption="Visualization of vector arithmetic performing analogy tasks (king - man + woman = queen)"
        />
        <Text size="sm" mt="md">
          Vector arithmetic operations in embedding space can capture semantic relationships like gender, tense, and more.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Embedding Space Visualization
        </Title>
        <Image
          src="/assets/data-science-practice/module8/word_embeddings.png"
          alt="Word Embedding Space Visualization"
          caption="2D projection of word embeddings showing semantic clustering"
        />
        <Text size="sm" mt="md">
          In this visualization, semantically related words cluster together, and meaningful relationships emerge as geometric patterns in the space.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">
          Word2Vec
        </Title>
        <Text mb="md">
          Word2Vec, introduced by Mikolov et al. (2013), learns word embeddings through shallow neural networks optimized to predict words from context or vice versa.
        </Text>
        <Text mb="md">
          Reference: <a href="https://arxiv.org/abs/1301.3781" target="_blank" rel="noopener noreferrer">Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)</a>
        </Text>
        <Text mt="md">
          Word2Vec has two distinct architectures:
        </Text>
        <List spacing="sm" mt="sm">
          <List.Item><strong>CBOW (Continuous Bag of Words)</strong>: Predicts a target word given its context words</List.Item>
          <List.Item><strong>Skip-gram</strong>: Predicts context words given a target word</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Word2Vec Architectures
        </Title>
        <Image
          src="/assets/data-science-practice/module8/word2vec-architectures.png"
          alt="Word2Vec Architectures"
          caption="Comparison of CBOW and Skip-gram architectures"
        />
        <Text size="sm" mt="md">
          Both architectures use shallow neural networks but differ in their prediction objectives.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Skip-gram Architecture
        </Title>
        <Text mb="md">
          <strong>Input</strong>:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>One-hot encoded target word vector <InlineMath math="\mathbf{x} \in \{0,1\}^{|V|}" /></List.Item>
          <List.Item>Context window size <InlineMath math="c" /> (typically 5-10)</List.Item>
        </List>

        <Text mb="md">
          <strong>Parameters</strong>:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Input embedding matrix <InlineMath math="\mathbf{W} \in \mathbb{R}^{|V| \times d}" /> (target word embeddings)</List.Item>
          <List.Item>Output embedding matrix <InlineMath math="\mathbf{W'} \in \mathbb{R}^{d \times |V|}" /> (context word embeddings)</List.Item>
          <List.Item>Dimensionality <InlineMath math="d" /> of the embedding space (typically 100-300)</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Skip-gram Forward Pass
        </Title>
        <Text mb="md">
          The forward pass consists of two steps:
        </Text>
        <Text mb="md">
          1. Embedding lookup: Extract the embedding for the target word
        </Text>
        <BlockMath math="\mathbf{h} = \mathbf{W}^T\mathbf{x}" />
        <Text mb="md">
          2. For each position in the context window, compute probability distribution over vocabulary:
        </Text>
        <BlockMath math="\mathbf{y} = \text{softmax}(\mathbf{W'h})" />
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Skip-gram Training Objective
        </Title>
        <Text mb="md">
          The Skip-gram model maximizes the log probability of context words given a target word:
        </Text>
        <BlockMath math="\mathcal{L} = \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j}|w_t)" />
        <Text mb="md">
          Where:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item><InlineMath math="T" /> is the total number of words in the corpus</List.Item>
          <List.Item><InlineMath math="c" /> is the context window size</List.Item>
          <List.Item><InlineMath math="P(w_{t+j}|w_t)" /> is modeled using the softmax function</List.Item>
        </List>
        <BlockMath math="P(w_j|w_i) = \frac{\exp(\mathbf{v}_{w_i}^T \mathbf{v}'_{w_j})}{\sum_{w \in V} \exp(\mathbf{v}_{w_i}^T \mathbf{v}'_w)}" />
        <Text>
          Where <InlineMath math="\mathbf{v}_{w_i}" /> is the input embedding of word <InlineMath math="w_i" /> and <InlineMath math="\mathbf{v}'_{w_j}" /> is the output embedding of word <InlineMath math="w_j" />.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Negative Sampling Illustration
        </Title>
        <Image
          src="/assets/data-science-practice/module8/negative-sampling.png"
          alt="Negative Sampling in Word2Vec"
          caption="Visualization of negative sampling: distinguishing true context words from random samples"
        />
        <Text size="sm" mt="md">
          Negative sampling approximates the softmax by training to distinguish true context words from random negatives.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Skip-gram Optimizations
        </Title>
        <Text mb="md">
          Computing the full softmax is computationally expensive for large vocabularies. Two main optimization techniques:
        </Text>
        <Text mb="md">
          <strong>1. Negative Sampling</strong>: Train to distinguish correct context words from randomly sampled "negative" words:
        </Text>
        <BlockMath math="\log \sigma(\mathbf{v}_{w_i}^T \mathbf{v}'_{w_j}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} [\log \sigma(-\mathbf{v}_{w_i}^T \mathbf{v}'_{w_k})]" />
        <Text mb="md">
          Where <InlineMath math="K" /> is the number of negative samples (typically 5-20) and <InlineMath math="P_n(w) \propto f(w)^{3/4}" /> where <InlineMath math="f(w)" /> is word frequency.
        </Text>
        <Text mt="md">
          <strong>2. Hierarchical Softmax</strong>: Uses a binary tree structure to reduce complexity from <InlineMath math="O(|V|)" /> to <InlineMath math="O(\log |V|)" />.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Skip-gram vs CBOW
        </Title>
        <Image
          src="/assets/data-science-practice/module8/skipgram-vs-cbow.png"
          alt="Skip-gram vs CBOW Comparison"
          caption="Detailed comparison of Skip-gram and CBOW prediction mechanisms"
        />
        <Text size="sm" mt="md">
          Skip-gram predicts context from target words, while CBOW predicts target words from averaged context.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          CBOW Architecture
        </Title>
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
          Same embedding matrices as Skip-gram: <InlineMath math="\mathbf{W} \in \mathbb{R}^{|V| \times d}" /> and <InlineMath math="\mathbf{W'} \in \mathbb{R}^{d \times |V|}" />
        </Text>

        <Text mb="md">
          <strong>Forward Pass</strong>:
        </Text>
        <Text mb="md">
          1. Average context word embeddings:
        </Text>
        <BlockMath math="\mathbf{h} = \frac{1}{2c}\sum_{i=1}^{2c} \mathbf{W}^T\mathbf{x}_i" />
        <Text mb="md">
          2. Predict target word:
        </Text>
        <BlockMath math="\mathbf{y} = \text{softmax}(\mathbf{W'h})" />
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          CBOW Training Objective
        </Title>
        <Text mb="md">
          The CBOW model maximizes the log probability of the target word given the context:
        </Text>
        <BlockMath math="\mathcal{L} = \sum_{t=1}^{T} \log P(w_t|w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})" />
        <Text mt="md" mb="md">
          <strong>Output Representation</strong>:
        </Text>
        <Text mb="md">
          After training, the input matrix <InlineMath math="\mathbf{W}" /> is typically used as the final word embedding matrix. Each row <InlineMath math="\mathbf{W}_i" /> represents the embedding of word <InlineMath math="i" /> in the vocabulary.
        </Text>
        <Text>
          Word similarity is computed using cosine similarity:
        </Text>
        <BlockMath math="\text{similarity}(w_i, w_j) = \frac{\mathbf{v}_{w_i} \cdot \mathbf{v}_{w_j}}{||\mathbf{v}_{w_i}|| \cdot ||\mathbf{v}_{w_j}||}" />
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Context Window Illustration
        </Title>
        <Image
          src="/assets/data-science-practice/module8/context-window.png"
          alt="Context Window in Word Embeddings"
          caption="Illustration of how context windows slide over text in Word2Vec and GloVe"
        />
        <Text size="sm" mt="md">
          The context window determines which words are considered neighbors for learning semantic relationships.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Word2Vec Implementation Example
        </Title>
        <CodeBlock
          language="python"
          code={`import gensim.downloader as api`}
        />
        <CodeBlock
          language="python"
          code={`# Load pre-trained Word2Vec embeddings
word2vec_model = api.load("word2vec-google-news-300")`}
        />
        <CodeBlock
          language="python"
          code={`# Get vector for a word
vector = word2vec_model["language"]  # 300-dimensional`}
        />
        <CodeBlock
          language="python"
          code={`# Find similar words
similar = word2vec_model.most_similar("language", topn=3)
print(similar)
# [('languages', 0.72), ('linguistic', 0.70), ...]`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Word2Vec Training Process
        </Title>
        <Image
          src="/assets/data-science-practice/module8/word2vec-training.png"
          alt="Word2Vec Training Process"
          caption="Visualization of Word2Vec training with sliding context windows"
        />
        <Text size="sm" mt="md">
          Word2Vec trains by sliding a context window over text, learning to predict contextual relationships.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">
          GloVe (Global Vectors)
        </Title>
        <Text mb="md">
          GloVe, introduced by Pennington et al. (2014), combines count-based and prediction-based approaches by training on global word-word co-occurrence statistics.
        </Text>
        <Text mb="md">
          Reference: <a href="https://nlp.stanford.edu/pubs/glove.pdf" target="_blank" rel="noopener noreferrer">GloVe: Global Vectors for Word Representation (Pennington et al., 2014)</a>
        </Text>
        <Text mt="md">
          Unlike Word2Vec which learns from local context windows in a sliding manner, GloVe aggregates global co-occurrence statistics across the entire corpus first, then learns embeddings to explain these statistics.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          GloVe Co-occurrence Matrix Visualization
        </Title>
        <Image
          src="/assets/data-science-practice/module8/glove-cooccurrence-matrix.png"
          alt="GloVe Co-occurrence Matrix"
          caption="Example of word-word co-occurrence matrix construction in GloVe"
        />
        <Text size="sm" mt="md">
          GloVe first constructs a co-occurrence matrix capturing how often words appear together across the corpus.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          GloVe Co-occurrence Matrix
        </Title>
        <Text mb="md">
          <strong>Preprocessing</strong>:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Build a co-occurrence matrix <InlineMath math="X" /> where <InlineMath math="X_{ij}" /> represents how often word <InlineMath math="i" /> appears in the context of word <InlineMath math="j" /></List.Item>
          <List.Item>Define a context window size (typically 5-10 words)</List.Item>
          <List.Item>Optionally apply distance weighting: words closer to the target contribute more</List.Item>
        </List>

        <Text mb="md">
          <strong>Parameters</strong>:
        </Text>
        <List spacing="sm">
          <List.Item>Word vectors <InlineMath math="\mathbf{w}_i \in \mathbb{R}^d" /> for each word <InlineMath math="i" /></List.Item>
          <List.Item>Context vectors <InlineMath math="\tilde{\mathbf{w}}_j \in \mathbb{R}^d" /> for each word <InlineMath math="j" /></List.Item>
          <List.Item>Word biases <InlineMath math="b_i" /> and context biases <InlineMath math="\tilde{b}_j" /></List.Item>
          <List.Item>Dimensionality <InlineMath math="d" /> (typically 100-300)</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          GloVe Training Objective
        </Title>
        <Text mb="md">
          GloVe minimizes the following weighted least squares cost function:
        </Text>
        <BlockMath math="J = \sum_{i,j=1}^{|V|} f(X_{ij})(\mathbf{w}_i^T\tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2" />
        <Text mb="md">
          The weighting function <InlineMath math="f" /> prevents rare co-occurrences from being overweighted:
        </Text>
        <BlockMath math="f(x) = \begin{cases}
          (x/x_{\max})^{\alpha} & \text{if } x < x_{\max} \\
          1 & \text{otherwise}
          \end{cases}" />
        <Text>
          Typically, <InlineMath math="\alpha = 0.75" /> and <InlineMath math="x_{\max} = 100" />.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          GloVe Key Insight
        </Title>
        <Text mb="md">
          The core insight of GloVe is that ratios of co-occurrence probabilities can encode meaning.
        </Text>
        <Text mb="md">
          For example, in a corpus about ice and steam, the ratio <InlineMath math="P_{ik}/P_{jk}" /> will be:
        </Text>
        <List spacing="sm" mt="sm">
          <List.Item>Large if word <InlineMath math="k" /> is related to ice (<InlineMath math="i" />) but not steam (<InlineMath math="j" />)</List.Item>
          <List.Item>Small if word <InlineMath math="k" /> is related to steam but not ice</List.Item>
          <List.Item>Close to 1 if word <InlineMath math="k" /> is related to both or neither</List.Item>
        </List>
        <Text mt="md">
          GloVe encodes these ratios as vector differences, making the embedding space capture semantic relationships naturally.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          GloVe Probability Ratios
        </Title>
        <Image
          src="/assets/data-science-practice/module8/glove-probability-ratios.png"
          alt="GloVe Probability Ratios"
          caption="Visualization of how probability ratios encode semantic relationships in GloVe"
        />
        <Text size="sm" mt="md">
          The ice-steam example showing how probability ratios distinguish related vs unrelated words.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          GloVe Implementation Example
        </Title>
        <CodeBlock
          language="python"
          code={`from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors`}
        />
        <CodeBlock
          language="python"
          code={`# Convert GloVe format to Word2Vec format
glove_input = "glove.6B.100d.txt"
word2vec_output = "glove.6B.100d.word2vec.txt"
glove2word2vec(glove_input, word2vec_output)`}
        />
        <CodeBlock
          language="python"
          code={`# Load the converted embeddings
glove_model = KeyedVectors.load_word2vec_format(word2vec_output)`}
        />
        <CodeBlock
          language="python"
          code={`# Get vector and find similar words
vector = glove_model["language"]  # 100-dimensional
similar = glove_model.most_similar("language", topn=3)`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="md">
          FastText
        </Title>
        <Text mb="md">
          FastText, developed by Facebook Research (Bojanowski et al., 2017), extends Word2Vec by representing words as bags of character n-grams, allowing it to capture morphological information and handle out-of-vocabulary words.
        </Text>
        <Text mb="md">
          Reference: <a href="https://arxiv.org/abs/1607.04606" target="_blank" rel="noopener noreferrer">Enriching Word Vectors with Subword Information (Bojanowski et al., 2017)</a>
        </Text>
        <Text mt="md">
          Key innovation: Instead of learning embeddings for whole words only, FastText learns embeddings for character n-grams, then represents words as sums of their n-gram embeddings.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          FastText Subword Embeddings
        </Title>
        <Image
          src="/assets/data-science-practice/module8/fasttext-subword-embeddings.png"
          alt="FastText Subword Embeddings"
          caption="Visualization of how FastText decomposes words into character n-grams"
        />
        <Text size="sm" mt="md">
          FastText represents words as bags of character n-grams, enabling better handling of morphologically rich languages.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          FastText N-gram Representation
        </Title>
        <Text mb="md">
          Each word is represented as a bag of character n-grams plus the word itself.
        </Text>
        <Text mb="md">
          For example, with n-grams of length 3-6, the word "where" becomes:
        </Text>
        <List spacing="sm" mt="sm">
          <List.Item>The word itself: "where"</List.Item>
          <List.Item>Character 3-grams: "&lt;wh", "whe", "her", "ere", "re&gt;"</List.Item>
          <List.Item>Character 4-grams: "&lt;whe", "wher", "here", "ere&gt;"</List.Item>
          <List.Item>Character 5-grams: "&lt;wher", "where", "here&gt;"</List.Item>
          <List.Item>Character 6-grams: "&lt;where", "where&gt;"</List.Item>
        </List>
        <Text mt="md" size="sm">
          Where "&lt;" and "&gt;" represent word boundary markers.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          FastText Architecture
        </Title>
        <Text mb="md">
          <strong>Parameters</strong>:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Embedding matrix <InlineMath math="\mathbf{Z} \in \mathbb{R}^{N \times d}" /> for all n-grams in the corpus</List.Item>
          <List.Item><InlineMath math="N" /> is the total number of unique n-grams plus words</List.Item>
          <List.Item><InlineMath math="d" /> is the embedding dimension (typically 100-300)</List.Item>
        </List>

        <Text mb="md">
          <strong>Word Representation</strong>:
        </Text>
        <Text mb="md">
          A word <InlineMath math="w" /> is represented as the sum of its n-gram embeddings:
        </Text>
        <BlockMath math="\mathbf{v}_w = \sum_{g \in G_w} \mathbf{z}_g" />
        <Text>
          Where <InlineMath math="G_w" /> is the set of n-grams in <InlineMath math="w" /> (including the word itself) and <InlineMath math="\mathbf{z}_g" /> is the vector for n-gram <InlineMath math="g" />.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          FastText Training Objective
        </Title>
        <Text mb="md">
          FastText can use either CBOW or Skip-gram architecture from Word2Vec, with the same loss functions but with word representations as sums of n-gram vectors.
        </Text>
        <Text mb="md">
          For Skip-gram:
        </Text>
        <BlockMath math="\mathcal{L} = \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j}|w_t)" />
        <Text mb="md">
          Where:
        </Text>
        <BlockMath math="P(w_j|w_i) = \frac{\exp(\mathbf{v}_{w_i}^T \mathbf{v}'_{w_j})}{\sum_{w \in V} \exp(\mathbf{v}_{w_i}^T \mathbf{v}'_w)}" />
        <Text>
          But now <InlineMath math="\mathbf{v}_{w_i} = \sum_{g \in G_{w_i}} \mathbf{z}_g" /> is the sum of n-gram vectors.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          FastText Advantages
        </Title>
        <List spacing="md">
          <List.Item>
            <strong>Handling Out-of-Vocabulary Words</strong>: Even for unseen words, FastText can generate embeddings by summing the embeddings of their character n-grams.
          </List.Item>
          <List.Item>
            <strong>Morphological Awareness</strong>: Words with similar morphological structures (e.g., "play", "plays", "playing") will share many n-grams and thus have similar embeddings.
          </List.Item>
          <List.Item>
            <strong>Rare Word Representation</strong>: Rare words benefit from sharing subcomponents with more common words.
          </List.Item>
        </List>
        <Text mt="md">
          Like Word2Vec, FastText uses negative sampling to efficiently approximate the softmax during training.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          FastText OOV Handling
        </Title>
        <Image
          src="/assets/data-science-practice/module8/fasttext-oov-handling.png"
          alt="FastText Out-of-Vocabulary Handling"
          caption="Example of FastText generating embeddings for unseen words using character n-grams"
        />
        <Text size="sm" mt="md">
          FastText can handle out-of-vocabulary words by composing embeddings from learned character n-grams.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          FastText Implementation Example
        </Title>
        <CodeBlock
          language="python"
          code={`from gensim.models import FastText`}
        />
        <CodeBlock
          language="python"
          code={`# Train a FastText model
sentences = [["natural", "language", "processing"],
             ["deep", "learning", "models"]]
model = FastText(sentences, vector_size=100, window=5)`}
        />
        <CodeBlock
          language="python"
          code={`# Vector for word in training data
vector = model.wv["language"]  # 100-dimensional`}
        />
        <CodeBlock
          language="python"
          code={`# Even for unseen words, FastText generates embeddings
unseen_vector = model.wv["multilingualism"]
similar = model.wv.most_similar("multilingualism", topn=3)
print(similar)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Embedding Methods Comparison
        </Title>
        <Table striped highlightOnHover mt="md">
          <thead>
            <tr>
              <th>Method</th>
              <th>Approach</th>
              <th>OOV Handling</th>
              <th>Complexity</th>
              <th>Best Use Case</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Word2Vec</strong></td>
              <td>Local context prediction</td>
              <td>No</td>
              <td>O(n) with neg. sampling</td>
              <td>Large corpora, semantic tasks</td>
            </tr>
            <tr>
              <td><strong>GloVe</strong></td>
              <td>Global co-occurrence</td>
              <td>No</td>
              <td>Depends on matrix size</td>
              <td>Analogy tasks, global statistics</td>
            </tr>
            <tr>
              <td><strong>FastText</strong></td>
              <td>Subword n-grams</td>
              <td>Yes</td>
              <td>O(n) with neg. sampling</td>
              <td>Morphologically rich languages, rare words</td>
            </tr>
          </tbody>
        </Table>
        <Text size="sm" mt="md">
          All three methods produce static embeddings: each word has a single fixed vector regardless of context.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Embedding Methods Comparison Visual
        </Title>
        <Image
          src="/assets/data-science-practice/module8/embedding-methods-comparison.png"
          alt="Comparison of Word2Vec, GloVe, and FastText"
          caption="Side-by-side comparison of the three major embedding approaches"
        />
        <Text size="sm" mt="md">
          Each method has distinct strengths: Word2Vec for semantic relationships, GloVe for global statistics, FastText for morphology.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Contextual vs Static Embeddings
        </Title>
        <Text mb="md">
          The embedding methods we've covered (Word2Vec, GloVe, FastText) produce <strong>static embeddings</strong>: each word has a single fixed vector regardless of context.
        </Text>
        <Text mb="md">
          For example, "bank" has the same embedding in both:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>"I went to the bank to deposit money" (financial institution)</List.Item>
          <List.Item>"I sat on the river bank" (land alongside water)</List.Item>
        </List>
        <Text mb="md">
          <strong>Contextual embeddings</strong> (from transformer models like BERT) generate different vectors for the same word based on its context. These will be covered in the transformer architectures section.
        </Text>
        <Text>
          Static embeddings remain valuable for their efficiency and effectiveness in many NLP tasks, particularly when context disambiguation is not critical.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Static vs Contextual Embeddings
        </Title>
        <Image
          src="/assets/data-science-practice/module8/static-vs-contextual.png"
          alt="Static vs Contextual Embeddings"
          caption="Comparison showing how static embeddings assign one vector per word vs contextual embeddings adapting to context"
        />
        <Text size="sm" mt="md">
          Static embeddings provide fixed representations, while contextual embeddings adapt based on surrounding words.
        </Text>
      </div>
    </>
  );
};

export default Embeddings;
