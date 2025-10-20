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
          <List.Item>Dense representations (d dimensions instead of vocabulary size)</List.Item>
          <List.Item>Learned from data rather than manually designed</List.Item>
          <List.Item>Semantic similarity correlates with vector distance</List.Item>
          <List.Item>Support arithmetic operations that capture analogies</List.Item>
        </List>
      </div>


      <div data-slide>
        <Title order={3} mb="md">
          WordLearned Embeddings
        </Title>
        <Text mb="md">
          Learned embeddings create dense vector representations by training on large text corpora, capturing semantic and syntactic relationships between words. Unlike one-hot encodings or simple count-based methods, these models learn to position words in a continuous vector space where semantic relationships are preserved as geometric properties.
        </Text>

      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Word Embeddings Visualization
        </Title>
        <Image
          src="/assets/data-science-practice/module8/embed.png"
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
                        <Image
          src="/assets/data-science-practice/module8/cosine.webp"
        />
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
          src="/assets/data-science-practice/module8/semantic-clustering.png"
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
          src="/assets/data-science-practice/module8/word2vec.png"
          alt="Word2Vec Architectures"
          caption="Comparison of CBOW and Skip-gram architectures"
        />
        <Text size="sm" mt="md">
          Skip-gram predicts context from target words, while CBOW predicts target words from averaged context.
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
          Contextual vs Static Embeddings
        </Title>
        <Text mb="md">
          The embedding methods (Word2Vec, GloVe, FastText) produce <strong>static embeddings</strong>: each word has a single fixed vector regardless of context.
        </Text>
        <Text mb="md">
          For example, "bank" has the same embedding in both:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>"I went to the bank to deposit money" (financial institution)</List.Item>
          <List.Item>"I sat on the river bank" (land alongside water)</List.Item>
        </List>
        <Text mb="md">
          <strong>Contextual embeddings</strong> (from transformer models like BERT) generate different vectors for the same word based on its context.
        </Text>
        <Text>
          Static embeddings remain valuable for their efficiency and effectiveness in many NLP tasks, particularly when context disambiguation is not critical.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">
          Contextual Embeddings
        </Title>
        <Text mb="md">
          Contextual embeddings are generated by transformer-based models that process entire sequences to produce context-aware representations. Unlike static embeddings, the same word receives different vectors depending on surrounding words.
        </Text>
        <Text mb="md">
          For a sequence <InlineMath math="[w_1, w_2, ..., w_n]" />, a transformer model produces hidden states <InlineMath math="[\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_n]" /> where each <InlineMath math="\mathbf{h}_i" /> encodes word <InlineMath math="w_i" /> in its context.
        </Text>
        <Text>
          Key advantage: The vector for "bank" in "river bank" will differ from "bank" in "deposit money", capturing the semantic distinction.
        </Text>
                      <Image
                src="/assets/data-science-practice/module8/contextual_embed.png"
                alt="Contextual Embeddings"
                caption="Comparison of Contextual and Static Embeddings"
              />
      </div>


      <div data-slide>
        <Title order={3} mb="md">
          Word-Level Contextual Embeddings
        </Title>
        <Text mb="md">
          To extract the contextual embedding for a specific word at position <InlineMath math="i" /> in the sequence:
        </Text>
        <BlockMath math="\mathbf{h}_i = \text{outputs.last\_hidden\_state}[:, i, :]" />
        <Text mb="md">
          This gives you a context-aware representation of that word. For example:
        </Text>
        <CodeBlock
          code={`token_idx = 2  # Word is at index 2
token_embedding = outputs.last_hidden_state[:, token_idx, :]  # Shape: (1, 768)`}
          language="python"
        />
        <Text mt="md">
          The embedding at position 2 represents that word in the context of the entire sentence, unlike static embeddings which are context-independent.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Sequence-Level Contextual Embeddings
        </Title>
        <Text mb="md">
          For many tasks, we need a single vector representing an entire sequence rather than individual words. This requires pooling strategies.
        </Text>
        <Text>
          Three main strategies exist for converting token-level hidden states into sentence or sequence embeddings:
        </Text>
        <List spacing="sm" mt="md">
          <List.Item><strong>CLS Token Pooling</strong>: Use a special token's representation (BERT)</List.Item>
          <List.Item><strong>Last Token Pooling</strong>: Use the final token's representation (GPT)</List.Item>
          <List.Item><strong>Mean Pooling</strong>: Average all token representations</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          BERT: CLS Token Pooling
        </Title>
        <Text mb="md">
          BERT (Bidirectional Encoder Representations from Transformers) uses a special <InlineMath math="\text{[CLS]}" /> token prepended to every sequence. The final hidden state of this token serves as the sequence representation.
        </Text>
        <BlockMath math="\mathbf{h}_{\text{sequence}} = \mathbf{h}_{\text{[CLS]}}" />
        <Text mb="md">
          The <InlineMath math="\text{[CLS]}" /> token's representation aggregates information from all tokens in the sequence through self-attention mechanisms.
        </Text>
        <Text>
          This approach is particularly effective for classification tasks, as BERT is often fine-tuned with a classification head on top of the <InlineMath math="\text{[CLS]}" /> token.
        </Text>
                <Image
          src="/assets/data-science-practice/module8/cls.png"
        />
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          GPT: Last Token Pooling
        </Title>
        <Text mb="md">
          GPT (Generative Pre-trained Transformer) uses causal (unidirectional) attention. The last token's hidden state accumulates information from all previous tokens in the sequence.
        </Text>
        <BlockMath math="\mathbf{h}_{\text{sequence}} = \mathbf{h}_n" />
        <Text mb="md">
          where <InlineMath math="n" /> is the sequence length. This final hidden state represents the entire sequence context.
        </Text>
        <Text>
          Since GPT processes text left-to-right, the last token has attended to all preceding tokens, making it a natural choice for sequence representation.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Mean Pooling
        </Title>
        <Text mb="md">
          Mean pooling averages the hidden states of all tokens in the sequence to create a single representation vector.
        </Text>
        <BlockMath math="\mathbf{h}_{\text{sequence}} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i" />
        <Text mb="md">
          This approach treats all tokens equally and is model-agnostic, working with any transformer architecture.
        </Text>
        <Text>
          Mean pooling often performs well for semantic similarity tasks and is commonly used with sentence embedding models like Sentence-BERT.
        </Text>
      </div>


      <div data-slide>
        <Title order={3} mb="md">
          Implementation with Hugging Face
        </Title>
        <Text mb="md">
          Import the necessary components from the transformers library:
        </Text>
        <CodeBlock
          code={`from transformers import AutoTokenizer, AutoModel
import torch`}
          language="python"
        />
        <Text mt="md" mb="sm">
          Load a pre-trained BERT model and its tokenizer:
        </Text>
        <CodeBlock
          code={`model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)`}
          language="python"
        />
        <Text mt="md">
          The tokenizer converts text to token IDs, while the model produces contextual embeddings.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Getting CLS Token Embeddings
        </Title>
        <Text mb="md">
          Tokenize input text and pass it through the model:
        </Text>
        <CodeBlock
          code={`text = "Natural language processing is fascinating"
inputs = tokenizer(text, return_tensors="pt")`}
          language="python"
        />
        <Text mt="md" mb="sm">
          Extract the CLS token embedding from model outputs:
        </Text>
        <CodeBlock
          code={`outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0, :]`}
          language="python"
        />
        <Text mt="md">
          The CLS token is always at position 0, so <InlineMath math="\text{[:, 0, :]}" /> extracts its embedding across all batches and dimensions.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Implementing Mean Pooling
        </Title>
        <Text mb="md">
          Define a function to compute mean pooling with attention mask:
        </Text>
        <CodeBlock
          code={`def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)`}
          language="python"
        />
        <Text mt="md">
          This function weights each token by its attention mask value (1 for real tokens, 0 for padding) and computes the average.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Applying Mean Pooling
        </Title>
        <Text mb="md">
          Use the mean pooling function with model outputs:
        </Text>
        <CodeBlock
          code={`outputs = model(**inputs)
mean_embedding = mean_pooling(outputs, inputs['attention_mask'])`}
          language="python"
        />
        <Text mt="md" mb="sm">
          Normalize the embeddings for cosine similarity:
        </Text>
        <CodeBlock
          code={`from torch.nn.functional import normalize
normalized_embedding = normalize(mean_embedding, p=2, dim=1)`}
          language="python"
        />
        <Text mt="md">
          Normalization ensures embeddings have unit length, making cosine similarity equivalent to dot product.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Comparing Sentences with Embeddings
        </Title>
        <Text mb="md">
          Compute embeddings for multiple sentences:
        </Text>
        <CodeBlock
          code={`sentences = [
    "The cat sits on the mat",
    "A feline rests on the rug",
    "Python is a programming language"
]
inputs = tokenizer(sentences, padding=True, return_tensors="pt")`}
          language="python"
        />
        <Text mt="md" mb="sm">
          Calculate pairwise cosine similarities:
        </Text>
        <CodeBlock
          code={`outputs = model(**inputs)
embeddings = normalize(mean_pooling(outputs, inputs['attention_mask']), p=2, dim=1)
similarities = torch.mm(embeddings, embeddings.transpose(0, 1))`}
          language="python"
        />
                        <Image
          src="/assets/data-science-practice/module8/Heatmap-cosine.png"
        />
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Sentence Transformers Library
        </Title>
        <Text mb="md">
          For production use, the sentence-transformers library provides optimized models specifically trained for semantic similarity:
        </Text>
        <CodeBlock
          code={`from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')`}
          language="python"
        />
        <Text mt="md" mb="sm">
          Generate embeddings with a single method call:
        </Text>
        <CodeBlock
          code={`sentences = ["This is a sentence", "This is another one"]
embeddings = model.encode(sentences)`}
          language="python"
        />
        <Text mt="md">
          These models are fine-tuned on sentence pairs to produce embeddings optimized for semantic similarity tasks.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">
          Training Sentence Embedding Models
        </Title>
        <Text mb="md">
          Sentence embedding models achieve high quality through supervised training on labeled sentence pairs. The training process optimizes embeddings to reflect semantic similarity.
        </Text>
        <Text mb="md">
          Training data consists of sentence pairs with similarity scores:
        </Text>
        <CodeBlock
          code={`training_data = [
    ("The dog is running", "A canine is jogging", 0.85),
    ("I love pizza", "The sky is blue", 0.10),
    ("Machine learning is powerful", "AI is transformative", 0.75)
]`}
          language="python"
        />
        <Text mt="md">
          Each tuple contains two sentences and their semantic similarity score ranging from 0 (completely unrelated) to 1 (semantically identical).
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Training Loop: Learning Semantic Similarity
        </Title>
        <Text mb="md">
          The model learns by minimizing the difference between predicted and actual similarity:
        </Text>
        <CodeBlock
          code={`for sentence1, sentence2, similarity_score in training_data:
    # Get embeddings using mean pooling
    emb1 = model.encode(sentence1)  # (384,)
    emb2 = model.encode(sentence2)  # (384,)`}
          language="python"
        />
        <Text mt="md" mb="sm">
          Compute similarity and update model parameters:
        </Text>
        <CodeBlock
          code={`    # Compute cosine similarity
    predicted_similarity = cosine_similarity(emb1, emb2)

    # Loss: difference between predicted and actual
    loss = (predicted_similarity - similarity_score) ** 2

    # Update model to make embeddings more accurate
    loss.backward()`}
          language="python"
        />
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Why This Training Produces Quality Embeddings
        </Title>
        <Text mb="md">
          The training objective forces the model to learn meaningful semantic representations:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Similar sentences are pushed closer in embedding space</List.Item>
          <List.Item>Dissimilar sentences are pushed further apart</List.Item>
          <List.Item>The model learns to ignore surface-level differences (synonyms, paraphrases)</List.Item>
          <List.Item>Semantic meaning becomes encoded in geometric proximity</List.Item>
        </List>
        <Text mb="md">
          After thousands of training examples, the model develops a robust understanding of semantic similarity that generalizes to new sentences.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Mean Pooling as Default for Sentence Embeddings
        </Title>
        <Text mb="md">
          When using <InlineMath math="\text{model.encode()}" /> in sentence transformers, mean pooling is applied by default to convert token-level embeddings into a single sentence vector.
        </Text>
        <Text mb="md">
          The process inside <InlineMath math="\text{encode()}" />:
        </Text>
        <CodeBlock
          code={`# Tokenize sentence
tokens = tokenizer(sentence, return_tensors="pt")

# Get token-level embeddings
outputs = model(**tokens)  # Shape: (1, num_tokens, 384)

# Apply mean pooling (default)
sentence_embedding = mean_pooling(outputs, tokens['attention_mask'])  # Shape: (1, 384)`}
          language="python"
        />
        <Text mt="md">
          Mean pooling averages all token representations while accounting for padding, producing a fixed-size vector that captures the entire sentence meaning.
        </Text>
      </div>


    </>
  );
};

export default Embeddings;
