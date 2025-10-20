import React from 'react';
import { Title, Text, List, Stack, Image, Flex } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';

export default function RAG() {
  return (
    <div>
      <div data-slide>
        <Title order={1} mb="lg">Retrieval-Augmented Generation (RAG)</Title>

        <Text size="lg" mb="md">
          Large Language Models have demonstrated remarkable capabilities across diverse applications including code generation, email composition, text summarization, and creative writing.
        </Text>

        <Text size="md" mb="md">
          However, LLMs face fundamental limitations:
        </Text>

        <List spacing="md" mb="xl">
          <List.Item>
            <Text weight={500} mb="xs">Hallucinations</Text>
            <Text size="sm">LLMs can generate plausible-sounding but factually incorrect information when they lack knowledge about a topic.</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">Outdated Knowledge</Text>
            <Text size="sm">Training data has a cutoff date, making models unaware of recent events or updated documentation.</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">No Access to Specific Data</Text>
            <Text size="sm">LLMs cannot access private documents, company databases, or any data not included in their training set.</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">Context Length Limitations</Text>
            <Text size="sm">Despite increasing context windows, LLMs lose performance when processing very large amounts of text.</Text>
          </List.Item>
        </List>
        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/rag0.png"
            alt="RAG architecture overview"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
        <Text size="md" c="dimmed">
          Retrieval-Augmented Generation addresses these limitations by combining LLMs with external knowledge retrieval systems.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">From Keyword Search to Semantic Retrieval</Title>

        <Text size="md" mb="md">
          Traditional databases use keyword-based search, matching exact words or patterns. However, as we explored in the embeddings section, semantic similarity provides a more powerful approach.
        </Text>

        <Text size="md" mb="md">
          Consider these queries that should retrieve similar documents but use different terminology:
        </Text>

        <List spacing="xs" mb="md">
          <List.Item>"How do neural networks learn?"</List.Item>
          <List.Item>"What is the training process for deep learning models?"</List.Item>
          <List.Item>"Explain backpropagation in AI systems"</List.Item>
        </List>

        <Text size="md" mb="md">
          Embeddings convert text into vector representations where semantically similar content has similar vectors, enabling contextual rather than lexical matching. RAG leverages this by creating a vector database where documents are stored as embeddings, enabling similarity-based retrieval.
        </Text>
        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/vector-search.jpeg"
            alt="RAG architecture overview"
            style={{ maxWidth: "70%", height: "auto" }}
          />
        </Flex>

      </div>

      <div data-slide>
        <Title order={2} mb="md">RAG Architecture Overview</Title>

        <Text size="md" mb="md">
          RAG enhances LLM responses by retrieving relevant information from external sources before generation. Instead of relying solely on the model's internal knowledge, we first retrieve relevant documents, then provide them as context to the LLM.
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/raggif.gif"
            alt="RAG architecture overview"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>

        <Title order={3} size="h4" mb="sm">Indexing Phase (Offline)</Title>
        <List spacing="xs" mb="lg" type="ordered">
          <List.Item>Split documents into chunks</List.Item>
          <List.Item>Generate embeddings for each chunk</List.Item>
          <List.Item>Store embeddings in a vector database with metadata</List.Item>
        </List>

        <Title order={3} size="h4" mb="sm">Retrieval Phase (Online)</Title>
        <List spacing="xs" type="ordered">
          <List.Item>User submits a query</List.Item>
          <List.Item>Query is embedded using the same model</List.Item>
          <List.Item>Vector database returns top-k most similar chunks</List.Item>
          <List.Item>Retrieved chunks are added to the LLM prompt</List.Item>
          <List.Item>LLM generates response based on retrieved context</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Component 1: Embedding Models</Title>

        <Text size="md" mb="md">
          The embedding model determines the quality of semantic search. The choice involves trade-offs between performance, cost, and retrieval quality.
        </Text>

        <Title order={3} size="h4" mb="sm">Popular Embedding Models</Title>
        <List spacing="sm" mb="lg">
          <List.Item>
            <Text weight={500}>text-embedding-3-small (OpenAI)</Text>
            <Text size="sm">1536 dimensions, balanced performance</Text>
          </List.Item>
          <List.Item>
            <Text weight={500}>Sentence-BERT</Text>
            <Text size="sm">384-768 dimensions, open-source</Text>
          </List.Item>
          <List.Item>
            <Text weight={500}>embed-v3 (Cohere)</Text>
            <Text size="sm">1024 dimensions, optimized for retrieval</Text>
          </List.Item>
        </List>

        <Title order={3} size="h4" mb="sm">Embedding Dimensions</Title>
        <Text size="md" mb="sm">
          Higher dimensions capture more nuanced semantic relationships but increase computational costs:
        </Text>
        <List spacing="xs" mb="md">
          <List.Item>Storage: <InlineMath>{'n_{docs} \\times n_{chunks} \\times d \\times 4\\text{ bytes}'}</InlineMath></List.Item>
          <List.Item>Query speed: More dimensions require more distance computations</List.Item>
          <List.Item>Retrieval quality: Better distinction between similar concepts</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="What is machine learning?"
)
embedding = response.data[0].embedding  # 1536 dimensions`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="md">Component 2: Chunking Strategy</Title>

        <Text size="md" mb="md">
          Documents must be split into chunks because embeddings have token limits and smaller chunks enable more precise retrieval.
        </Text>

        <Title order={3} size="h4" mb="sm">Key Chunking Considerations</Title>
        <List spacing="sm" mb="lg">
          <List.Item>
            <Text weight={500}>Chunk size:</Text> Typical ranges are 200-500 tokens. Smaller chunks are more precise but may lack context; larger chunks provide more context but may be less relevant.
          </List.Item>
          <List.Item>
            <Text weight={500}>Overlap:</Text> Adding 10-20% overlap between chunks prevents information loss at boundaries.
          </List.Item>
          <List.Item>
            <Text weight={500}>Semantic boundaries:</Text> Split at paragraph or sentence boundaries rather than arbitrary character counts.
          </List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks`}
        />

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/chunk.png"
            alt="Chunking Strategy Visualization"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Component 3: Vector Database and Indexing</Title>

        <Text size="md" mb="md">
          Vector databases are specialized for storing and searching high-dimensional embeddings efficiently using similarity metrics.
        </Text>

        <Title order={3} size="h4" mb="sm">Similarity Search: Cosine Similarity</Title>
        <Text size="md" mb="sm">
          The most common distance metric measures the angle between vectors:
        </Text>

        <BlockMath>
          {'\\text{similarity}(q, d) = \\frac{q \\cdot d}{\\|q\\| \\|d\\|} = \\cos(\\theta)'}
        </BlockMath>

        <Text size="md" mt="md" mb="md">
          Values range from -1 (opposite) to 1 (identical). RAG systems retrieve the top-k documents with highest similarity scores.
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/indexing.webp"
            alt="Indexing Strategy Visualization"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>

        <CodeBlock
          language="python"
          code={`import faiss
import numpy as np

# Create index
dimension = 1536
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_matrix)

# Search
k = 5  # top 5 results
distances, indices = index.search(query_embedding, k)`}
        />

        <Title order={3} size="h4" mb="sm" mt="xl">Indexing Strategies for Scale</Title>
        <Text size="md" mb="md">
          When dealing with billions of vectors, computing similarity metrics against every vector becomes computationally prohibitive. A common indexing strategy uses clustering to reduce search space:
        </Text>

        <List spacing="sm" mb="md">
          <List.Item>
            <Text weight={500}>K-means clustering:</Text> Group vectors into clusters (e.g., 1000 clusters)
          </List.Item>
          <List.Item>
            <Text weight={500}>Two-stage search:</Text> First, compute distance to 1000 cluster centroids, then search within the nearest cluster
          </List.Item>
          <List.Item>
            <Text weight={500}>Efficiency gain:</Text> Instead of computing billions of distances, compute approximately 1000 distances twice
          </List.Item>
        </List>

        <Text size="md" c="dimmed">
          This approximate nearest neighbor approach trades a small amount of accuracy for massive computational savings, making billion-scale vector search practical.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Component 4: Prompt Engineering</Title>

        <Text size="md" mb="md">
          The final step combines retrieved documents with the user query in a carefully structured prompt.
        </Text>

        <Title order={3} size="h4" mb="sm">Prompt Structure</Title>
        <List spacing="xs" mb="lg" type="ordered">
          <List.Item>System instructions (behavior, grounding requirements)</List.Item>
          <List.Item>Retrieved documents as context</List.Item>
          <List.Item>User query</List.Item>
          <List.Item>Output format specification</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say so.

Context:
{retrieved_docs}

Question: {user_query}

Answer:"""

response = llm.generate(prompt)`}
        />

        <Text size="md" mt="lg" mb="sm">
          Advanced techniques include:
        </Text>
        <List spacing="xs">
          <List.Item>Instructing the model to cite sources</List.Item>
          <List.Item>Requesting confidence levels for claims</List.Item>
          <List.Item>Distinguishing retrieved information from general knowledge</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Complete RAG Example</Title>

        <Text size="md" mb="md">
          A minimal RAG implementation using Python:
        </Text>

        <CodeBlock
          language="python"
          code={`from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)`}
        />

        <Text size="md" mt="md" mb="sm">Create embeddings and vector database:</Text>

        <CodeBlock
          language="python"
          code={`# Create embeddings
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(chunks, embeddings)

# Retrieve and generate
query = "What is transfer learning?"
docs = vectordb.similarity_search(query, k=3)
context = "\\n".join([doc.page_content for doc in docs])
response = llm.generate(f"Context: {context}\\n\\nQuestion: {query}")`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="md">Evaluation Metrics</Title>

        <Text size="md" mb="md">
          RAG systems should be evaluated on both retrieval and generation quality independently.
        </Text>

        <Title order={3} size="h4" mb="sm">Retrieval Metrics</Title>
        <List spacing="xs" mb="lg">
          <List.Item><Text weight={500}>Precision:</Text> Proportion of retrieved documents that are relevant</List.Item>
          <List.Item><Text weight={500}>Recall:</Text> Proportion of relevant documents that are retrieved</List.Item>
          <List.Item><Text weight={500}>MRR:</Text> Mean Reciprocal Rank - ranking quality of results</List.Item>
        </List>

        <Title order={3} size="h4" mb="sm">Generation Metrics</Title>
        <List spacing="xs" mb="lg">
          <List.Item><Text weight={500}>Faithfulness:</Text> Does the answer align with retrieved documents?</List.Item>
          <List.Item><Text weight={500}>Answer relevance:</Text> Does it address the user query?</List.Item>
          <List.Item><Text weight={500}>Context utilization:</Text> Is retrieved information effectively used?</List.Item>
        </List>

        <Text size="md">
          Optimization strategies include adjusting chunk size, tuning k (number of retrieved documents), experimenting with embedding models, and refining prompt templates.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">NotebookLM</Title>

        <Text size="md" mb="md">
          Google's NotebookLM demonstrates a production RAG system for interacting with documents through natural language.
        </Text>

        <Text size="md" mb="md">
          Explore NotebookLM: <Text span component="a" href="https://notebooklm.google.com/" target="_blank" c="blue" td="underline">notebooklm.google.com</Text>
        </Text>

  
      </div>

    </div>
  );
}
