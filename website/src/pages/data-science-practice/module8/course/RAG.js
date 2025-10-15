import React from "react";
import { Container, Title, Text, Space, Image, List, Table, Flex } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const RAG = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={1} mb="md" id="rag">
          Retrieval-Augmented Generation (RAG)
        </Title>
        <Text>
          Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generation-based approaches to create
          more accurate, grounded, and trustworthy text generation systems. RAG enhances large language models (LLMs) by retrieving
          relevant information from external knowledge sources before generating responses.
        </Text>
        <Space h="md" />
        <Text>
          The power of RAG lies in its ability to separate knowledge (stored in external databases) from reasoning (performed by the LLM).
          This separation allows for dynamic knowledge updates without retraining the model and provides traceable sources for generated information.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="sm" id="architecture">
          RAG Architecture and Workflow
        </Title>
        <Text mb="md">
          RAG systems follow a multi-stage pipeline that combines retrieval and generation components. The architecture integrates
          a retrieval system with a generative language model to produce more informed outputs.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Core Components of RAG
        </Title>
        <Flex gap="xl" direction={{ base: 'column', md: 'row' }}>
          <div style={{ flex: 1 }}>
            <List>
              <List.Item>
                <Text fw={500}>Document Processing</Text>: Ingestion, chunking, embedding and indexing of knowledge base documents
              </List.Item>
              <List.Item>
                <Text fw={500}>Retriever</Text>: Responsible for finding relevant documents from the knowledge base given a query
              </List.Item>
              <List.Item>
                <Text fw={500}>Generator</Text>: Language model that produces text conditioned on the retrieved documents and query
              </List.Item>
              <List.Item>
                <Text fw={500}>Orchestrator</Text>: Controls the overall workflow, including query processing and post-processing
              </List.Item>
            </List>
          </div>
          <div style={{ flex: 1 }}>
            <Image
              src="/api/placeholder/640/360"
              alt="RAG Architecture Diagram"
            />
            <Text size="sm" c="dimmed" mt="xs">Basic RAG architecture showing document processing, retrieval, and generation components</Text>
          </div>
        </Flex>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="rag-workflow">
          RAG Workflow
        </Title>
        <Text mb="md">
          The typical RAG workflow consists of two main phases: offline indexing and online serving.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Offline Indexing Phase
        </Title>
        <List>
          <List.Item>
            <Text fw={500}>Document Collection</Text>: Gather text data from various sources (web pages, PDFs, databases)
          </List.Item>
          <List.Item>
            <Text fw={500}>Text Preprocessing</Text>: Clean, normalize, and prepare texts for processing
          </List.Item>
          <List.Item>
            <Text fw={500}>Chunking</Text>: Split documents into manageable chunks that balance context and retrieval granularity
          </List.Item>
          <List.Item>
            <Text fw={500}>Embedding Generation</Text>: Convert text chunks into vector embeddings using embedding models
          </List.Item>
          <List.Item>
            <Text fw={500}>Indexing</Text>: Store embeddings in a vector database optimized for similarity search
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Online Serving Phase
        </Title>
        <List type="ordered">
          <List.Item>
            <Text fw={500}>Query Processing</Text>: User query is processed and converted to an embedding
          </List.Item>
          <List.Item>
            <Text fw={500}>Retrieval</Text>: System searches the vector database for chunks most similar to the query
          </List.Item>
          <List.Item>
            <Text fw={500}>Context Augmentation</Text>: Retrieved chunks are formatted and added to the prompt
          </List.Item>
          <List.Item>
            <Text fw={500}>Generation</Text>: LLM generates a response based on the query and retrieved context
          </List.Item>
          <List.Item>
            <Text fw={500}>Post-processing</Text>: Optional filtering, fact-checking, or reformatting of the response
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Basic RAG Implementation
        </Title>
        <Text mb="md">Here's a basic RAG implementation with LangChain:</Text>
        <CodeBlock
          language="python"
          code={`from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter`}
        />
        <Space h="sm" />
        <Text size="sm" mb="sm">Load and split documents:</Text>
        <CodeBlock
          language="python"
          code={`# Load documents
loader = TextLoader("path/to/document.txt")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Basic RAG Implementation (continued)
        </Title>
        <Text size="sm" mb="sm">Create embeddings and vector database:</Text>
        <CodeBlock
          language="python"
          code={`# Create embeddings and store in vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embeddings)

# Initialize retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})`}
        />
        <Space h="sm" />
        <Text size="sm" mb="sm">Set up the RAG chain:</Text>
        <CodeBlock
          language="python"
          code={`# Load LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    model_kwargs={"temperature": 0.1, "max_length": 512}
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
result = qa_chain.run("What are the key components of a RAG system?")`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="sm" id="vector-databases">
          Vector Databases
        </Title>
        <Text mb="md">
          Vector databases are specialized database systems optimized for storing and retrieving high-dimensional vectors.
          In RAG systems, they serve as the foundation for efficient similarity search operations.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="vector-db-fundamentals">
          Vector Database Fundamentals
        </Title>
        <Title order={4} mb="sm">Key Features</Title>
        <List mb="lg">
          <List.Item><Text fw={500}>Vector Operations</Text>: Efficient computation of similarity metrics (cosine, Euclidean, dot product)</List.Item>
          <List.Item><Text fw={500}>Indexing Structures</Text>: Advanced indexing for fast approximate nearest neighbor (ANN) search</List.Item>
          <List.Item><Text fw={500}>Metadata Filtering</Text>: Support for filtering based on document metadata</List.Item>
          <List.Item><Text fw={500}>Scalability</Text>: Ability to handle millions or billions of vectors</List.Item>
          <List.Item><Text fw={500}>CRUD Operations</Text>: Support for adding, updating, and deleting vectors</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Popular Vector Databases
        </Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Database</Table.Th>
              <Table.Th>Type</Table.Th>
              <Table.Th>Key Features</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Pinecone</Table.Td>
              <Table.Td>Cloud-native</Table.Td>
              <Table.Td>Fully managed, serverless, high scalability</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Weaviate</Table.Td>
              <Table.Td>Open-source</Table.Td>
              <Table.Td>GraphQL API, multi-modal, hybrid search</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Chroma</Table.Td>
              <Table.Td>Open-source</Table.Td>
              <Table.Td>Lightweight, easy to use, Python-first</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Milvus</Table.Td>
              <Table.Td>Open-source</Table.Td>
              <Table.Td>Distributed architecture, high throughput</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Qdrant</Table.Td>
              <Table.Td>Open-source</Table.Td>
              <Table.Td>Rust-based, filtering, payload storage</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Vector Similarity Metrics
        </Title>
        <Text mb="md">
          The choice of similarity metric is crucial for retrieval performance. Each metric has different mathematical properties and use cases.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Cosine Similarity</Title>
        <Text mb="sm">
          Measures the cosine of the angle between two vectors, focusing on direction rather than magnitude.
        </Text>
        <BlockMath>{`\\cos(\\theta) = \\frac{A \\cdot B}{||A|| ||B||} = \\frac{\\sum_{i=1}^{n} A_i B_i}{\\sqrt{\\sum_{i=1}^{n} A_i^2} \\sqrt{\\sum_{i=1}^{n} B_i^2}}`}</BlockMath>
        <Space h="md" />
        <Text size="sm">Best for comparing document similarity regardless of document length.</Text>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Euclidean Distance</Title>
        <Text mb="sm">
          Measures the straight-line distance between two points in vector space.
        </Text>
        <BlockMath>{`d(A, B) = ||A - B|| = \\sqrt{\\sum_{i=1}^{n} (A_i - B_i)^2}`}</BlockMath>
        <Space h="md" />
        <Text size="sm">Useful when the magnitude of vectors matters and lower values indicate higher similarity.</Text>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Dot Product</Title>
        <Text mb="sm">
          Sum of the products of corresponding vector components, sensitive to both direction and magnitude.
        </Text>
        <BlockMath>{`A \\cdot B = \\sum_{i=1}^{n} A_i B_i`}</BlockMath>
        <Space h="md" />
        <Text size="sm">Effective for normalized vectors and computationally efficient.</Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Connecting to Vector Databases
        </Title>
        <Text mb="sm">Example with Chroma:</Text>
        <CodeBlock
          language="python"
          code={`from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_chroma = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# Query Chroma
docs = db_chroma.similarity_search("What is RAG?", k=3)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Connecting to Pinecone
        </Title>
        <CodeBlock
          language="python"
          code={`from langchain.vectorstores import Pinecone
import pinecone

# Initialize Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENV")
index_name = "langchain-demo"`}
        />
        <Space h="sm" />
        <Text size="sm" mb="sm">Create the index if it doesn't exist:</Text>
        <CodeBlock
          language="python"
          code={`if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384, metric="cosine")

# Load documents into Pinecone
db_pinecone = Pinecone.from_documents(documents, embeddings, index_name=index_name)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Connecting to Weaviate
        </Title>
        <CodeBlock
          language="python"
          code={`from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

# Start embedded Weaviate (for local testing)
client = weaviate.Client(embedded_options=EmbeddedOptions())`}
        />
        <Space h="sm" />
        <Text size="sm" mb="sm">Create a Weaviate vector store with metadata filtering:</Text>
        <CodeBlock
          language="python"
          code={`db_weaviate = Weaviate.from_documents(
    documents, embeddings, client=client, class_name="RAGDocuments"
)

# Advanced query with metadata filtering
results = db_weaviate.similarity_search_with_score(
    "What is RAG?", k=3, where_filter={"source": "textbook"}
)`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="sm" id="chunking">
          Document Chunking and Indexing
        </Title>
        <Text mb="md">
          Effective document processing is crucial for RAG system performance. This involves breaking down documents into
          appropriate chunks, embedding them, and indexing them for efficient retrieval.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="document-chunking">
          Document Chunking Strategies
        </Title>
        <Text mb="md">
          Chunking involves splitting large documents into smaller, semantically meaningful units that can be efficiently
          embedded and retrieved. The choice of chunking strategy significantly impacts retrieval quality.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Chunk Size Considerations
        </Title>
        <Flex gap="xl" direction={{ base: 'column', md: 'row' }}>
          <div style={{ flex: 1 }}>
            <Text fw={500} mb="xs">Small Chunks (100-500 tokens)</Text>
            <List>
              <List.Item><Text>More precise retrieval for specific information</Text></List.Item>
              <List.Item><Text>Lower context utilization by LLM</Text></List.Item>
              <List.Item><Text>Risk of losing broader context</Text></List.Item>
              <List.Item><Text>Good for factoid QA and specific details</Text></List.Item>
            </List>
          </div>
          <div style={{ flex: 1 }}>
            <Text fw={500} mb="xs">Large Chunks (1000+ tokens)</Text>
            <List>
              <List.Item><Text>Better preservation of context</Text></List.Item>
              <List.Item><Text>More comprehensive information</Text></List.Item>
              <List.Item><Text>Risk of diluting key information</Text></List.Item>
              <List.Item><Text>Good for summarization and complex reasoning</Text></List.Item>
            </List>
          </div>
        </Flex>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Chunk Overlap
        </Title>
        <Text mb="md">
          Overlap between chunks helps maintain context continuity and prevents information from being lost at chunk boundaries.
          Typical overlap ranges from 10-20% of the chunk size.
        </Text>
        <Flex gap="xl" direction={{ base: 'column', md: 'row' }}>
          <div style={{ flex: 1 }}>
            <Text fw={500} mb="xs">Benefits of Overlap</Text>
            <List>
              <List.Item><Text>Prevents loss of context at chunk boundaries</Text></List.Item>
              <List.Item><Text>Improves retrieval of information that spans chunks</Text></List.Item>
              <List.Item><Text>Provides redundancy for important information</Text></List.Item>
            </List>
          </div>
          <div style={{ flex: 1 }}>
            <Text fw={500} mb="xs">Considerations</Text>
            <List>
              <List.Item><Text>Increases storage and computational requirements</Text></List.Item>
              <List.Item><Text>Can lead to duplicate information in responses</Text></List.Item>
              <List.Item><Text>Requires balancing with chunk size</Text></List.Item>
            </List>
          </div>
        </Flex>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Chunking Methods
        </Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Method</Table.Th>
              <Table.Th>Description</Table.Th>
              <Table.Th>Best For</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Character-based</Table.Td>
              <Table.Td>Split by character count with overlap</Table.Td>
              <Table.Td>Simple texts, uniform content</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Token-based</Table.Td>
              <Table.Td>Split by token count (LLM tokens)</Table.Td>
              <Table.Td>Optimizing for LLM context windows</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Sentence-based</Table.Td>
              <Table.Td>Split at sentence boundaries</Table.Td>
              <Table.Td>Preserving semantic units</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Paragraph-based</Table.Td>
              <Table.Td>Split at paragraph boundaries</Table.Td>
              <Table.Td>Maintaining topical coherence</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Recursive</Table.Td>
              <Table.Td>Use multiple delimiters in hierarchical order</Table.Td>
              <Table.Td>Structured documents with headings/sections</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Semantic</Table.Td>
              <Table.Td>Split based on semantic similarity</Table.Td>
              <Table.Td>Content with varying topic density</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="document-indexing">
          Document Indexing Process
        </Title>
        <Text mb="md">
          The indexing process involves generating embeddings for each document chunk and storing them in a vector database
          for efficient retrieval.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Document Processing Pipeline
        </Title>
        <Flex gap="xl" direction={{ base: 'column', md: 'row' }}>
          <div style={{ flex: 1 }}>
            <Image
              src="/api/placeholder/640/480"
              alt="Document Processing Pipeline"
            />
            <Text size="sm" c="dimmed" mt="xs">Document processing pipeline for RAG systems</Text>
          </div>
          <div style={{ flex: 1 }}>
            <List type="ordered">
              <List.Item>
                <Text fw={500}>Document Collection</Text>
                <Text size="sm">Gather documents from various sources (web pages, PDFs, databases)</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Text Extraction & Cleaning</Text>
                <Text size="sm">Extract text, remove formatting, normalize content</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Chunking</Text>
                <Text size="sm">Split documents into appropriate-sized chunks</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Embedding Generation</Text>
                <Text size="sm">Convert chunks to vector embeddings using embedding models</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Metadata Enrichment</Text>
                <Text size="sm">Add metadata (source, date, categories) to enhance filtering</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Vector Storage</Text>
                <Text size="sm">Store embeddings and metadata in vector database</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Index Building</Text>
                <Text size="sm">Create indexing structures for efficient search</Text>
              </List.Item>
            </List>
          </div>
        </Flex>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Advanced Chunking Techniques
        </Title>
        <Text mb="sm">Import necessary text splitters:</Text>
        <CodeBlock
          language="python"
          code={`from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain.document_loaders import PyPDFLoader`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Different Chunking Approaches
        </Title>
        <Text mb="sm">Character-based chunking (simple approach):</Text>
        <CodeBlock
          language="python"
          code={`char_splitter = CharacterTextSplitter(
    separator="\\n\\n",
    chunk_size=1000,
    chunk_overlap=200
)
char_chunks = char_splitter.split_documents(pages)`}
        />
        <Space h="sm" />
        <Text mb="sm">Recursive character splitting (more sophisticated):</Text>
        <CodeBlock
          language="python"
          code={`recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\\n\\n", "\\n", ". ", ", ", " "],
    chunk_size=1000,
    chunk_overlap=200
)
recursive_chunks = recursive_splitter.split_documents(pages)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Token-Based Chunking
        </Title>
        <Text mb="sm">Token-based splitting (optimized for LLM token limits):</Text>
        <CodeBlock
          language="python"
          code={`token_splitter = TokenTextSplitter(
    model_name="gpt-3.5-turbo",
    chunk_size=500,  # In tokens, not characters
    chunk_overlap=50
)
token_chunks = token_splitter.split_documents(pages)`}
        />
        <Space h="sm" />
        <Text mb="sm">Advanced token splitting for sentence-transformers:</Text>
        <CodeBlock
          language="python"
          code={`st_splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=256,
    chunk_overlap=25
)
st_chunks = st_splitter.split_documents(pages)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Analyzing Chunk Quality
        </Title>
        <CodeBlock
          language="python"
          code={`import numpy as np

# Calculate token distribution statistics
chunk_lengths = [len(chunk.page_content.split()) for chunk in recursive_chunks]
print(f"Average chunk size (words): {np.mean(chunk_lengths):.1f}")
print(f"Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}")`}
        />
        <Space h="sm" />
        <Text mb="sm">Check for potential issues in chunks:</Text>
        <CodeBlock
          language="python"
          code={`problem_chunks = []
for i, chunk in enumerate(recursive_chunks):
    content = chunk.page_content
    if len(content) < 100:
        problem_chunks.append((i, "Too short", len(content)))
    elif not content.strip().endswith((".", "!", "?")):
        problem_chunks.append((i, "No sentence ending"))`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="sm" id="similarity-search">
          Similarity Search Algorithms
        </Title>
        <Text mb="md">
          Similarity search is the core retrieval operation in RAG systems. It involves finding the most relevant documents
          from a vector database given a query embedding.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Approximate Nearest Neighbor (ANN) Search
        </Title>
        <Text mb="md">
          As vector databases grow to millions or billions of vectors, exact nearest neighbor search becomes computationally
          prohibitive. Approximate Nearest Neighbor (ANN) algorithms provide efficient alternatives that trade a small amount
          of accuracy for significant performance gains.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Key ANN Algorithms
        </Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Algorithm</Table.Th>
              <Table.Th>Description</Table.Th>
              <Table.Th>Trade-offs</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Locality-Sensitive Hashing (LSH)</Table.Td>
              <Table.Td>Uses hash functions to map similar vectors to the same buckets</Table.Td>
              <Table.Td>Fast, but less accurate than tree-based methods</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Hierarchical Navigable Small World (HNSW)</Table.Td>
              <Table.Td>Multi-layered graph structure with skip connections</Table.Td>
              <Table.Td>Excellent recall/speed trade-off, high memory usage</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Product Quantization (PQ)</Table.Td>
              <Table.Td>Compresses vectors by splitting into subvectors and quantizing</Table.Td>
              <Table.Td>Memory efficient, lower precision</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Inverted File Index (IVF)</Table.Td>
              <Table.Td>Partitions vector space into clusters, search only relevant clusters</Table.Td>
              <Table.Td>Fast for high dimensions, sensitive to cluster quality</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Performance Metrics
        </Title>
        <List>
          <List.Item>
            <Text fw={500}>Recall@k</Text>
            <Text size="sm" mb="xs">
              The fraction of true nearest neighbors found among the top-k results.
            </Text>
            <BlockMath>{`Recall@k = \\frac{|\\text{Retrieved True Neighbors} \\cap \\text{Actual Top-k Neighbors}|}{|\\text{Actual Top-k Neighbors}|}`}</BlockMath>
          </List.Item>
          <List.Item mt="md">
            <Text fw={500}>Queries Per Second (QPS)</Text>
            <Text size="sm">
              The number of queries that can be processed per second.
            </Text>
          </List.Item>
          <List.Item mt="md">
            <Text fw={500}>Index Build Time</Text>
            <Text size="sm">
              Time required to build the search index.
            </Text>
          </List.Item>
          <List.Item mt="md">
            <Text fw={500}>Memory Usage</Text>
            <Text size="sm">
              RAM required for index storage and search operations.
            </Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="hybrid-search">
          Hybrid Search Techniques
        </Title>
        <Text mb="md">
          Hybrid search combines multiple retrieval approaches to improve overall retrieval quality.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Vector + Keyword Search</Title>
        <Text mb="md">
          Combines semantic similarity (vector) with lexical matching (keyword) for more comprehensive retrieval.
        </Text>
        <List>
          <List.Item>
            <Text fw={500}>BM25 + Vector Search</Text>
            <Text size="sm">Combines the classical BM25 algorithm with neural embeddings</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Sparse-Dense Fusion</Text>
            <Text size="sm">Merges results from sparse (keyword) and dense (embedding) retrieval</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Weighted Combinations</Text>
            <Text size="sm">Applies different weights to lexical and semantic components</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Re-ranking Approaches</Title>
        <Text mb="md">
          Uses a two-stage retrieval process: efficient first-pass retrieval followed by more
          sophisticated re-ranking of candidates.
        </Text>
        <List>
          <List.Item>
            <Text fw={500}>Cross-Encoder Re-ranking</Text>
            <Text size="sm">Uses attention between query and document for precise relevance scoring</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Ensemble Re-ranking</Text>
            <Text size="sm">Combines multiple models for better relevance assessment</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>LLM Re-ranking</Text>
            <Text size="sm">Uses LLM to directly evaluate and score document relevance</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Advanced Similarity Search Implementation
        </Title>
        <Text mb="sm">Using BM25 (keyword-based) retrieval alongside vector search:</Text>
        <CodeBlock
          language="python"
          code={`from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Base vector retriever
vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Ensemble Retriever
        </Title>
        <Text mb="sm">Create an ensemble retriever that combines both approaches:</Text>
        <CodeBlock
          language="python"
          code={`# Create an ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # Equal weighting
)

# Query the ensemble retriever
results = ensemble_retriever.get_relevant_documents(
    "What is transfer learning in NLP?"
)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Re-ranking with Cross-Encoder
        </Title>
        <Text mb="sm">Load a cross-encoder model for relevance ranking:</Text>
        <CodeBlock
          language="python"
          code={`from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Re-ranking Function
        </Title>
        <CodeBlock
          language="python"
          code={`def rerank_documents(query, docs, top_k=3):
    # Prepare inputs for cross-encoder
    pairs = [(query, doc.page_content) for doc in docs]

    # Tokenize and get model inputs
    features = tokenizer(pairs, padding=True, truncation=True,
                        return_tensors="pt", max_length=512)

    # Get relevance scores
    with torch.no_grad():
        scores = model(**features).logits.flatten()

    # Sort documents by score
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in doc_score_pairs[:top_k]]`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Contextual Compression
        </Title>
        <Text mb="sm">Using LLM to extract relevant parts from retrieved documents:</Text>
        <CodeBlock
          language="python"
          code={`from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation"
)`}
        />
        <Space h="sm" />
        <CodeBlock
          language="python"
          code={`# Create a compressor that extracts relevant information
compressor = LLMChainExtractor.from_llm(llm)

# Create a compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever
)

compressed_docs = compression_retriever.get_relevant_documents("What is transfer learning?")`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="sm" id="pipeline">
          Complete RAG Pipeline
        </Title>
        <Text mb="md">
          Building a production-ready RAG system involves integrating all the previously discussed components into a coherent pipeline.
          This section covers the end-to-end implementation of a RAG system.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="pipeline-components">
          Pipeline Components and Integration
        </Title>
        <Flex gap="xl" direction={{ base: 'column', md: 'row' }}>
          <div style={{ flex: 1 }}>
            <Title order={4} mb="sm">Key Pipeline Components</Title>
            <List>
              <List.Item>
                <Text fw={500}>Data Ingestion System</Text>
                <Text size="sm">Handles document collection, preprocessing, and updating</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Embedding Service</Text>
                <Text size="sm">Converts text to vector embeddings</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Vector Database</Text>
                <Text size="sm">Stores and indexes document embeddings</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Retriever System</Text>
                <Text size="sm">Implements similarity search and re-ranking</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Context Processor</Text>
                <Text size="sm">Formats retrieved documents for prompt construction</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>LLM Service</Text>
                <Text size="sm">Generates responses based on query and context</Text>
              </List.Item>
              <List.Item>
                <Text fw={500}>Orchestrator</Text>
                <Text size="sm">Coordinates workflow and handles error cases</Text>
              </List.Item>
            </List>
          </div>
          <div style={{ flex: 1 }}>
            <Image
              src="/api/placeholder/640/480"
              alt="RAG Pipeline Architecture"
            />
            <Text size="sm" c="dimmed" mt="xs">End-to-end RAG pipeline architecture</Text>
          </div>
        </Flex>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="prompt-engineering">
          RAG Prompt Engineering
        </Title>
        <Text mb="md">
          The design of prompts for RAG systems is crucial for effectively leveraging the retrieved context and
          generating accurate responses.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Prompt Components
        </Title>
        <List>
          <List.Item>
            <Text fw={500}>System Instructions</Text>
            <Text size="sm">Define the overall behavior and constraints for the LLM</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Context Introduction</Text>
            <Text size="sm">Explain how to use the provided context</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Retrieved Documents</Text>
            <Text size="sm">The actual retrieved content from the knowledge base</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>User Query</Text>
            <Text size="sm">The original question or instruction from the user</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Output Format Instructions</Text>
            <Text size="sm">Guidelines for how the response should be structured</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Example Prompt Template
        </Title>
        <CodeBlock
          language="text"
          code={`You are a helpful AI assistant that answers questions based on the provided context.

CONTEXT:
{retrieved_documents}

USER QUESTION:
{query}

Answer the user's question using only the information provided in the context above.
If the context doesn't contain the relevant information, say "I don't have enough
information to answer this question." Do not make up information.

Provide your answer in a clear, concise manner. Include relevant quotes from the
context where appropriate, marked with quotation marks.`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Advanced Prompt Strategies
        </Title>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Chain-of-Verification</Title>
        <Text mb="md">
          Instructs the LLM to critically evaluate the retrieved information before generating a response.
          The LLM first analyzes each retrieved passage for relevance and accuracy, then generates the final answer.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Step-by-Step Reasoning</Title>
        <Text mb="md">
          Explicitly guides the LLM to break down complex queries into sub-problems,
          reason through each sub-problem using the retrieved context, and then synthesize a final answer.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Source Attribution</Title>
        <Text mb="md">
          Requires the LLM to cite the specific documents or passages used for each part of the response,
          improving transparency and enabling verification.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Query Reformulation</Title>
        <Text mb="md">
          Instructs the LLM to rephrase or expand the original query to better match the knowledge base's content,
          improving retrieval quality before answering.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Complete RAG Pipeline Implementation
        </Title>
        <Text mb="sm">Set up logging and configure pipeline parameters:</Text>
        <CodeBlock
          language="python"
          code={`import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure pipeline parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 5`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Document Loading Function
        </Title>
        <CodeBlock
          language="python"
          code={`def load_documents(sources):
    """Load documents from multiple sources"""
    documents = []

    for source in sources:
        logger.info(f"Loading document from: {source}")
        try:
            if source.endswith('.txt'):
                loader = TextLoader(source)
            elif source.endswith('.pdf'):
                loader = PyPDFLoader(source)
            elif source.startswith('http'):
                loader = WebBaseLoader(source)
            else:
                logger.warning(f"Unsupported source type: {source}")
                continue

            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Error loading {source}: {str(e)}")

    return documents`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Document Processing Function
        </Title>
        <CodeBlock
          language="python"
          code={`def process_documents(documents):
    """Split documents into chunks and generate embeddings"""
    logger.info("Processing documents...")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\\n\\n", "\\n", ". ", ", ", " "]
    )
    chunks = text_splitter.split_documents(documents)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Document Processing (continued)
        </Title>
        <CodeBlock
          language="python"
          code={`    # Add metadata for tracking
    for i, chunk in enumerate(chunks):
        if 'source' not in chunk.metadata:
            chunk.metadata['source'] = f"document_{i}"
        chunk.metadata['chunk_id'] = i
        chunk.metadata['timestamp'] = datetime.now().isoformat()

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectordb`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          RAG Chain Setup
        </Title>
        <CodeBlock
          language="python"
          code={`def setup_rag_chain(vectordb):
    """Configure the RAG retrieval and generation chain"""
    # Configure retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # Set up LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL,
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Prompt Template for RAG Chain
        </Title>
        <CodeBlock
          language="python"
          code={`    template = """
    You are a helpful AI assistant that answers questions based on the provided context.

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    Answer the user's question using only the information provided in the context above.
    If the context doesn't contain the relevant information, say "I don't have enough
    information to answer this question." Do not make up information.
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Create the RAG Chain
        </Title>
        <CodeBlock
          language="python"
          code={`    # Create the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return rag_chain`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Query Execution Function
        </Title>
        <CodeBlock
          language="python"
          code={`def execute_query(rag_chain, query):
    """Execute a query against the RAG system"""
    logger.info(f"Executing query: {query}")

    try:
        start_time = datetime.now()
        result = rag_chain({"query": query})
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"Query processed in {processing_time:.2f} seconds")

        return {
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return {"error": str(e)}`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Main RAG Pipeline Function
        </Title>
        <CodeBlock
          language="python"
          code={`def run_rag_pipeline(sources, query):
    """Run the complete RAG pipeline"""
    # Load documents
    documents = load_documents(sources)

    # Process documents
    vectordb = process_documents(documents)

    # Setup RAG chain
    rag_chain = setup_rag_chain(vectordb)

    # Execute query
    result = execute_query(rag_chain, query)

    return result`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="sm" id="evaluation">
          RAG System Evaluation
        </Title>
        <Text mb="md">
          Evaluating RAG systems is essential for measuring performance, identifying bottlenecks, and ensuring
          the system meets quality requirements. This involves assessing both retrieval and generation components.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Retrieval Evaluation Metrics
        </Title>
        <List>
          <List.Item>
            <Text fw={500}>Precision@k</Text>
            <Text size="sm">Fraction of retrieved documents that are relevant</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Recall@k</Text>
            <Text size="sm">Fraction of relevant documents that are retrieved</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Mean Reciprocal Rank (MRR)</Text>
            <Text size="sm">Average of reciprocal ranks of first relevant documents</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Normalized Discounted Cumulative Gain (NDCG)</Text>
            <Text size="sm">Measures ranking quality with relevance grading</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Mean Average Precision (MAP)</Text>
            <Text size="sm">Mean of average precision across queries</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Generation Evaluation Metrics
        </Title>
        <List>
          <List.Item>
            <Text fw={500}>Faithfulness/Factuality</Text>
            <Text size="sm">Measures alignment with retrieved context</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Relevance</Text>
            <Text size="sm">Assesses response relevance to the query</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Correctness</Text>
            <Text size="sm">Verifies factual accuracy against ground truth</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Coherence</Text>
            <Text size="sm">Evaluates logical flow and readability</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Hallucination Rate</Text>
            <Text size="sm">Measures occurrence of generated content not in context</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="evaluation-methods">
          Evaluation Methods
        </Title>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Manual Evaluation
        </Title>
        <Text mb="md">
          Human evaluation provides in-depth assessment of RAG system outputs but is resource-intensive and difficult to scale.
        </Text>
        <Flex gap="xl" direction={{ base: 'column', md: 'row' }}>
          <div style={{ flex: 1 }}>
            <Text fw={500} mb="xs">Approaches</Text>
            <List>
              <List.Item><Text>Expert review of response accuracy</Text></List.Item>
              <List.Item><Text>Side-by-side comparison with baseline systems</Text></List.Item>
              <List.Item><Text>Human rating on multiple quality dimensions</Text></List.Item>
              <List.Item><Text>User satisfaction surveys</Text></List.Item>
            </List>
          </div>
          <div style={{ flex: 1 }}>
            <Text fw={500} mb="xs">Best Practices</Text>
            <List>
              <List.Item><Text>Use standardized evaluation rubrics</Text></List.Item>
              <List.Item><Text>Ensure diverse evaluator backgrounds</Text></List.Item>
              <List.Item><Text>Implement inter-rater reliability checks</Text></List.Item>
              <List.Item><Text>Balance qualitative and quantitative feedback</Text></List.Item>
            </List>
          </div>
        </Flex>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Automatic Evaluation
        </Title>
        <Text mb="md">
          Automatic evaluation methods enable large-scale assessment of RAG systems and provide consistent metrics for comparison.
        </Text>
        <Flex gap="xl" direction={{ base: 'column', md: 'row' }}>
          <div style={{ flex: 1 }}>
            <Text fw={500} mb="xs">Reference-Based Methods</Text>
            <List>
              <List.Item><Text>ROUGE (Recall-Oriented Understudy for Gisting Evaluation)</Text></List.Item>
              <List.Item><Text>BLEU (Bilingual Evaluation Understudy)</Text></List.Item>
              <List.Item><Text>METEOR (Metric for Evaluation of Translation with Explicit Ordering)</Text></List.Item>
              <List.Item><Text>BERTScore (contextual embedding similarity)</Text></List.Item>
            </List>
          </div>
          <div style={{ flex: 1 }}>
            <Text fw={500} mb="xs">LLM-as-a-Judge Methods</Text>
            <List>
              <List.Item><Text>LLM-based factuality checking</Text></List.Item>
              <List.Item><Text>Consistency evaluation with retrieved context</Text></List.Item>
              <List.Item><Text>Multi-dimensional quality assessment</Text></List.Item>
              <List.Item><Text>Question answering accuracy evaluation</Text></List.Item>
            </List>
          </div>
        </Flex>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Benchmarking Datasets
        </Title>
        <Text mb="md">
          Standardized datasets provide consistent evaluation across different RAG implementations.
        </Text>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Benchmark</Table.Th>
              <Table.Th>Focus Area</Table.Th>
              <Table.Th>Key Features</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>KILT</Table.Td>
              <Table.Td>Knowledge-Intensive Tasks</Table.Td>
              <Table.Td>Fact checking, entity linking, slot filling, QA</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>MMLU</Table.Td>
              <Table.Td>Multitask Language Understanding</Table.Td>
              <Table.Td>Professional and academic subjects with multiple-choice questions</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>NaturalQuestions</Table.Td>
              <Table.Td>Open-domain QA</Table.Td>
              <Table.Td>Real Google search queries with Wikipedia answers</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>HotpotQA</Table.Td>
              <Table.Td>Multi-hop reasoning</Table.Td>
              <Table.Td>Questions requiring information from multiple documents</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>RAGAs</Table.Td>
              <Table.Td>RAG-specific evaluation</Table.Td>
              <Table.Td>Context relevance, faithfulness, answer relevance metrics</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          RAG Evaluation Implementation
        </Title>
        <Text mb="sm">Import evaluation tools and load dataset:</Text>
        <CodeBlock
          language="python"
          code={`from langchain.evaluation import load_evaluator, EvaluatorType
from datasets import load_dataset
import numpy as np
import pandas as pd

# Load a question-answering dataset for evaluation
dataset = load_dataset("nq_open", split="validation[:100]")`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Retrieval Evaluation Function
        </Title>
        <CodeBlock
          language="python"
          code={`def evaluate_retrieval(retriever, dataset, ground_truth_docs):
    """Evaluate retriever performance"""
    results = {"precision": [], "recall": [], "reciprocal_rank": []}

    for example in dataset:
        query = example["question"]
        relevant_doc_ids = ground_truth_docs.get(query, [])

        retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_ids = [doc.metadata.get("id") for doc in retrieved_docs]

        relevant_retrieved = set(relevant_doc_ids).intersection(set(retrieved_ids))

        # Calculate metrics
        precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
        recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Retrieval Evaluation (continued)
        </Title>
        <CodeBlock
          language="python"
          code={`        # Reciprocal Rank
        rr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_doc_ids:
                rr = 1 / (i + 1)
                break

        results["precision"].append(precision)
        results["recall"].append(recall)
        results["reciprocal_rank"].append(rr)

    return {
        "avg_precision": np.mean(results["precision"]),
        "avg_recall": np.mean(results["recall"]),
        "mean_reciprocal_rank": np.mean(results["reciprocal_rank"])
    }`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          LLM-based Response Evaluation
        </Title>
        <CodeBlock
          language="python"
          code={`def evaluate_generation(rag_chain, dataset):
    """Evaluate the quality of generated responses"""
    faithfulness_evaluator = load_evaluator(
        EvaluatorType.LABELED_CRITERIA,
        criteria={"faithfulness": "Does the response only contain information
                                   from the retrieved context?"}
    )

    results = []
    for example in dataset:
        query = example["question"]
        reference_answer = example["answer"]

        response = rag_chain({"query": query})
        generated_answer = response["result"]
        context = "\\n".join([doc.page_content for doc in response["source_documents"]])`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Response Evaluation (continued)
        </Title>
        <CodeBlock
          language="python"
          code={`        # Evaluate faithfulness
        faithfulness_result = faithfulness_evaluator.evaluate_strings(
            prediction=generated_answer,
            reference=context,
            input=query
        )

        results.append({
            "query": query,
            "generated_answer": generated_answer,
            "faithfulness_score": faithfulness_result["normalized_score"]
        })

    df = pd.DataFrame(results)
    return {"avg_faithfulness": df["faithfulness_score"].mean()}, df`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Analyzing Failure Cases
        </Title>
        <CodeBlock
          language="python"
          code={`def analyze_failure_cases(evaluation_results, threshold=0.5):
    """Analyze cases where the RAG system performed poorly"""
    df = pd.DataFrame(evaluation_results)

    # Identify low-performing cases
    low_faithfulness = df[df["faithfulness_score"] < threshold]

    # Analyze retrieval issues
    retrieval_issues = []
    for _, row in low_faithfulness.iterrows():
        context = row["context"]
        if len(context) < 100:
            retrieval_issues.append({
                "query": row["query"],
                "issue": "Insufficient relevant context retrieved"
            })

    return {"retrieval_issues": retrieval_issues}`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="sm" id="prompt-techniques">
          Advanced Prompt Techniques
        </Title>
        <Text mb="md">
          Effective prompt design is crucial for maximizing the performance of RAG systems. This section covers
          advanced prompt techniques specifically tailored for retrieval-augmented generation.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="retrieval-focused">
          Retrieval-Focused Prompt Techniques
        </Title>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Query Transformation</Title>
        <Text mb="md">
          Techniques that modify or expand the original query to improve retrieval performance.
        </Text>
        <List>
          <List.Item>
            <Text fw={500}>Hypothetical Document Embeddings (HyDE)</Text>
            <Text size="sm">Uses an LLM to generate a hypothetical ideal document for the query, then embeds that document</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Query Expansion</Text>
            <Text size="sm">Adds related terms to the original query to broaden the retrieval scope</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Query Decomposition</Text>
            <Text size="sm">Breaks complex queries into simpler sub-queries for more targeted retrieval</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Self-Query</Text>
            <Text size="sm">Uses the LLM to generate structured queries with filters for the retrieval system</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Multi-Step Retrieval</Title>
        <Text mb="md">
          Techniques that employ multiple retrieval rounds to refine and improve results.
        </Text>
        <List>
          <List.Item>
            <Text fw={500}>Retrieval-Augmented Retrieval</Text>
            <Text size="sm">Uses an initial retrieval to inform a second, more targeted retrieval</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Recursive Retrieval</Text>
            <Text size="sm">Repeatedly refines queries based on previously retrieved information</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Step-Back Prompting</Text>
            <Text size="sm">First retrieves high-level concepts, then detailed information</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>FLARE (Forward-Looking Active REtrieval)</Text>
            <Text size="sm">Generates parts of the response, pauses to retrieve more context when needed</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm" id="generation-focused">
          Generation-Focused Prompt Techniques
        </Title>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Context Integration</Title>
        <Text mb="md">
          Techniques for effectively incorporating retrieved information into the generation prompt.
        </Text>
        <List>
          <List.Item>
            <Text fw={500}>Few-Shot Context Examples</Text>
            <Text size="sm">Include examples of ideal responses based on context</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Context Distillation</Text>
            <Text size="sm">Use an LLM to summarize or extract key points from retrieved documents</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Context Ranking</Text>
            <Text size="sm">Present most relevant context snippets first in the prompt</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Context Segmentation</Text>
            <Text size="sm">Clearly demarcate different context sources with headers and metadata</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={4} mb="sm">Response Constraints</Title>
        <Text mb="md">
          Techniques that guide the generation process for better accuracy and reliability.
        </Text>
        <List>
          <List.Item>
            <Text fw={500}>Chain-of-Verification</Text>
            <Text size="sm">Instruct the LLM to verify information from context before generating a response</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Citation Requirements</Text>
            <Text size="sm">Require specific citations to context passages for each claim</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Structured Output Format</Text>
            <Text size="sm">Define specific JSON, XML, or markdown formats for responses</Text>
          </List.Item>
          <List.Item>
            <Text fw={500}>Confidence Annotation</Text>
            <Text size="sm">Request confidence levels for different parts of the response</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Advanced RAG Prompt Template
        </Title>
        <Text mb="md">
          This template incorporates multiple advanced techniques for high-quality RAG responses.
        </Text>
        <CodeBlock
          language="text"
          code={`You are an AI assistant that provides accurate, helpful responses based on the provided context.

CONTEXT INFORMATION:
{context}

USER QUERY:
{query}

Follow these steps carefully:

1) CONTEXT ANALYSIS:
   - Identify which parts of the context are relevant to the query
   - Note any conflicting information or gaps in the provided context
   - Determine if the context contains sufficient information to answer the query

2) RESPONSE FORMULATION:
   - Answer the query using ONLY information from the context
   - If the context lacks necessary information, acknowledge this limitation
   - Present information in a logical, structured manner
   - Include direct quotes from the context where appropriate
   - Cite the specific source for each key piece of information

3) VERIFICATION:
   - Review your response to ensure all claims are supported by the context
   - Remove any statements not directly supported by the provided information
   - Assess your confidence in the accuracy of the response`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Implementing HyDE (Hypothetical Document Embeddings)
        </Title>
        <CodeBlock
          language="python"
          code={`from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

def setup_hyde_retriever(vectordb, llm):
    """Create a HyDE retriever"""
    hyde_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Write a passage that directly answers this question:
{question}

Make the passage detailed, informative, and factually correct."""
    )

    hyde_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=LLMChain(llm=llm, prompt=hyde_prompt),
        base_embeddings=vectordb.embeddings
    )

    return vectordb.as_retriever(search_kwargs={"k": 5})`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Implementing Self-Query Retriever
        </Title>
        <CodeBlock
          language="python"
          code={`from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

def setup_self_query_retriever(vectordb, llm):
    """Create a retriever that constructs its own filters"""
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The source document or URL",
            type="string",
        ),
        AttributeInfo(
            name="date",
            description="The date the document was published",
            type="string",
        ),
    ]`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Self-Query Retriever (continued)
        </Title>
        <CodeBlock
          language="python"
          code={`    document_content_description = "Academic and technical information about NLP"

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectordb,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True
    )

    return retriever`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Query Decomposition
        </Title>
        <CodeBlock
          language="python"
          code={`def decompose_query(llm, query):
    """Break down a complex query into simpler sub-queries"""
    decomposition_prompt = PromptTemplate(
        input_variables=["query"],
        template="""Break this question down into 2-4 simpler sub-questions.

Original Question: {query}

Output the sub-questions as a numbered list."""
    )

    decomposition_chain = LLMChain(llm=llm, prompt=decomposition_prompt)
    result = decomposition_chain.run(query=query)

    # Parse sub-queries
    sub_queries = []
    for line in result.strip().split("\\n"):
        if line and (line.startswith("- ") or any(line.startswith(f"{i}.") for i in range(1, 10))):
            clean_line = line.split(".", 1)[-1].strip() if "." in line else line[2:].strip()
            sub_queries.append(clean_line)

    return sub_queries`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Chain-of-Verification Prompt
        </Title>
        <CodeBlock
          language="python"
          code={`def create_verification_chain(llm, retriever):
    """Create a chain that verifies information before generating"""
    verification_prompt = PromptTemplate(
        template="""You are a careful assistant that answers questions based on context.

CONTEXT:
{context}

USER QUESTION:
{question}

Follow these steps:

1. VERIFICATION PHASE:
   - Analyze each piece of retrieved information for relevance
   - Check for contradictions or inconsistencies
   - Evaluate the reliability and completeness

2. ANSWER FORMULATION:
   - Provide a direct answer to the user's question
   - Only include information explicitly supported by the context
   - Cite specific parts of the context

ANSWER:""",
        input_variables=["context", "question"]
    )`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          Verification Chain (continued)
        </Title>
        <CodeBlock
          language="python"
          code={`    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": verification_prompt},
        reduce_k_below_max_tokens=True
    )

    return chain`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          FLARE: Forward-Looking Active Retrieval
        </Title>
        <Text mb="sm">Implement a FLARE-like approach:</Text>
        <CodeBlock
          language="python"
          code={`def flare_rag_generation(llm, retriever, query, max_iterations=3):
    """Generation is interleaved with retrieval"""
    # Initial retrieval
    context = "\\n\\n".join([doc.page_content
                            for doc in retriever.get_relevant_documents(query)])

    system_prompt = """As you generate your response, identify when you need
more information and write [SEARCH: your specific search query] on a new line.
After all necessary searches, provide your final answer."""

    partial_response = ""
    search_results = [context]`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="sm">
          FLARE Implementation (continued)
        </Title>
        <CodeBlock
          language="python"
          code={`    for i in range(max_iterations):
        # Construct prompt with all context
        current_prompt = f"{system_prompt}\\n\\nCONTEXT:\\n"
        for idx, ctx in enumerate(search_results):
            current_prompt += f"--- Context {idx+1} ---\\n{ctx}\\n\\n"

        current_prompt += f"QUESTION: {query}\\n\\nRESPONSE SO FAR:\\n{partial_response}\\n\\n"

        # Get next part of response
        next_part = llm(current_prompt)

        # Check if a search is requested
        if "[SEARCH:" in next_part:
            before_search, after_search = next_part.split("[SEARCH:", 1)
            search_query = after_search.split("]", 1)[0]

            partial_response += before_search

            # Perform additional retrieval
            new_docs = retriever.get_relevant_documents(search_query.strip())
            new_context = "\\n\\n".join([doc.page_content for doc in new_docs])
            search_results.append(new_context)
        else:
            partial_response += next_part
            break

    return partial_response`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="sm" id="further-reading">
          Further Reading
        </Title>
        <Text fw={500} mb="sm">Research Papers</Text>
        <List mb="lg">
          <List.Item>
            <Text>Lewis, P., et al. (2021). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"</Text>
          </List.Item>
          <List.Item>
            <Text>Gao, L., et al. (2023). "Precise Zero-Shot Dense Retrieval without Relevance Labels"</Text>
          </List.Item>
          <List.Item>
            <Text>Trivedi, H., et al. (2023). "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"</Text>
          </List.Item>
        </List>
        <Text fw={500} mb="sm">Tutorials and Resources</Text>
        <List>
          <List.Item>
            <Text>LangChain RAG Documentation: https://python.langchain.com/docs/use_cases/question_answering/</Text>
          </List.Item>
          <List.Item>
            <Text>Hugging Face RAG Guide: https://huggingface.co/docs/transformers/tasks/retrieval_augmented_generation</Text>
          </List.Item>
          <List.Item>
            <Text>Pinecone Learning Center: https://www.pinecone.io/learn/</Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2} mb="sm">
          Key Takeaways
        </Title>
        <Flex gap="xl" direction={{ base: 'column', md: 'row' }}>
          <div style={{ flex: 1 }}>
            <List>
              <List.Item>
                <Text>RAG systems combine the power of retrieval and generation to produce more accurate and grounded responses</Text>
              </List.Item>
              <List.Item>
                <Text>Effective document processing, including chunking and embedding, is crucial for retrieval quality</Text>
              </List.Item>
              <List.Item>
                <Text>Vector databases enable efficient similarity search for finding relevant information</Text>
              </List.Item>
              <List.Item>
                <Text>Advanced retrieval techniques like hybrid search and re-ranking improve result quality</Text>
              </List.Item>
            </List>
          </div>
          <div style={{ flex: 1 }}>
            <List>
              <List.Item>
                <Text>Prompt engineering is essential for effective context integration and response generation</Text>
              </List.Item>
              <List.Item>
                <Text>Comprehensive evaluation frameworks help identify and address system weaknesses</Text>
              </List.Item>
              <List.Item>
                <Text>Advanced techniques like query transformation and multi-step retrieval can significantly enhance RAG performance</Text>
              </List.Item>
              <List.Item>
                <Text>RAG systems provide a practical approach to grounding LLM outputs in verified information</Text>
              </List.Item>
            </List>
          </div>
        </Flex>
      </div>
    </Container>
  );
};

export default RAG;
