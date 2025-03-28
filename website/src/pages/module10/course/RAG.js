import React from "react";
import {
  Container,
  Title,
  Text,
  Space,
  Card,
  Image,
  Grid,
  List,
  Table,
  Accordion,
  Code,
  Tabs,
  Box,
  Anchor,
  ThemeIcon,
} from "@mantine/core";
import { IconInfoCircle, IconBulb } from "@tabler/icons-react";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";


const RAG = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md" id="rag">
        Retrieval-Augmented Generation (RAG)
      </Title>
      <Text>
        Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generation-based approaches to create
        more accurate, grounded, and trustworthy text generation systems. RAG enhances large language models (LLMs) by retrieving
        relevant information from external knowledge sources before generating responses.
      </Text>

      <Space h="xl" />

      {/* 1. RAG Architecture and Workflow */}
      <Title order={2} mb="sm" id="architecture">
        RAG Architecture and Workflow
      </Title>
      <Text mb="md">
        RAG systems follow a multi-stage pipeline that combines retrieval and generation components. The architecture integrates
        a retrieval system with a generative language model to produce more informed outputs.
      </Text>

      <Card shadow="sm" p="lg" radius="md" mb="lg" withBorder>
        <Title order={3} mb="sm">
          Core Components of RAG
        </Title>
        <Grid>
          <Grid.Col span={{ base: 12, md: 6 }}>
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
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Image
              src="/api/placeholder/640/360"
              alt="RAG Architecture Diagram"
              caption="Basic RAG architecture showing document processing, retrieval, and generation components"
            />
          </Grid.Col>
        </Grid>
      </Card>

      <Title order={3} mb="sm" id="rag-workflow">
        RAG Workflow
      </Title>
      <Text mb="md">
        The typical RAG workflow consists of two main phases: offline indexing and online serving.
      </Text>

      <Accordion variant="separated" mb="xl">
        <Accordion.Item value="offline">
          <Accordion.Control>
            <Text fw={500}>Offline Indexing Phase</Text>
          </Accordion.Control>
          <Accordion.Panel>
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
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="online">
          <Accordion.Control>
            <Text fw={500}>Online Serving Phase</Text>
          </Accordion.Control>
          <Accordion.Panel>
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
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <Card shadow="sm" p="lg" radius="md" mb="lg" withBorder>
        <Grid>
          <Grid.Col span={{ base: 12, md: 3 }}>
            <ThemeIcon size="xl" color="blue" radius="md">
              <IconBulb size={24} />
            </ThemeIcon>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 9 }}>
            <Text fw={500} mb="xs">Key Insight</Text>
            <Text size="sm">
              The power of RAG lies in its ability to separate knowledge (stored in external databases) from reasoning (performed by the LLM).
              This separation allows for dynamic knowledge updates without retraining the model and provides traceable sources for generated information.
            </Text>
          </Grid.Col>
        </Grid>
      </Card>

      <CodeBlock
        language="python"
        code={`
# Basic RAG implementation with LangChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# 1. Load documents
loader = TextLoader("path/to/document.txt")
documents = loader.load()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and store in vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embeddings)

# 4. Initialize retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 5. Load LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    model_kwargs={"temperature": 0.1, "max_length": 512}
)

# 6. Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Other options: "map_reduce", "refine", "map_rerank"
    retriever=retriever,
)

# 7. Query the system
query = "What are the key components of a RAG system?"
result = qa_chain.run(query)
print(result)
`}
      />

      <Space h="xl" />

      {/* 2. Vector Databases */}
      <Title order={2} mb="sm" id="vector-databases">
        Vector Databases
      </Title>
      <Text mb="md">
        Vector databases are specialized database systems optimized for storing and retrieving high-dimensional vectors.
        In RAG systems, they serve as the foundation for efficient similarity search operations.
      </Text>

      <Title order={3} mb="sm" id="vector-db-fundamentals">
        Vector Database Fundamentals
      </Title>

      <Grid mb="lg">
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" h="100%" withBorder>
            <Title order={4} mb="sm">Key Features</Title>
            <List>
              <List.Item><Text fw={500}>Vector Operations</Text>: Efficient computation of similarity metrics (cosine, Euclidean, dot product)</List.Item>
              <List.Item><Text fw={500}>Indexing Structures</Text>: Advanced indexing for fast approximate nearest neighbor (ANN) search</List.Item>
              <List.Item><Text fw={500}>Metadata Filtering</Text>: Support for filtering based on document metadata</List.Item>
              <List.Item><Text fw={500}>Scalability</Text>: Ability to handle millions or billions of vectors</List.Item>
              <List.Item><Text fw={500}>CRUD Operations</Text>: Support for adding, updating, and deleting vectors</List.Item>
            </List>
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" h="100%" withBorder>
            <Title order={4} mb="sm">Popular Vector Databases</Title>
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
          </Card>
        </Grid.Col>
      </Grid>

      <Card shadow="sm" p="lg" radius="md" mb="lg" withBorder>
        <Title order={3} mb="sm">Vector Similarity Metrics</Title>
        <Text mb="md">
          The choice of similarity metric is crucial for retrieval performance. Each metric has different mathematical properties and use cases.
        </Text>
          <Grid mb="md">
            <Grid.Col span={{ base: 12, md: 4 }}>
              <Card shadow="xs" p="md" radius="md" withBorder>
                <Title order={4} mb="xs">Cosine Similarity</Title>
                <Text size="sm" mb="sm">
                  Measures the cosine of the angle between two vectors, focusing on direction rather than magnitude.
                </Text>
                <BlockMath>{`\\cos(\\theta) = \\frac{A \\cdot B}{||A|| ||B||} = \\frac{\\sum_{i=1}^{n} A_i B_i}{\\sqrt{\\sum_{i=1}^{n} A_i^2} \\sqrt{\\sum_{i=1}^{n} B_i^2}}`}</BlockMath>
              </Card>
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 4 }}>
              <Card shadow="xs" p="md" radius="md" withBorder>
                <Title order={4} mb="xs">Euclidean Distance</Title>
                <Text size="sm" mb="sm">
                  Measures the straight-line distance between two points in vector space.
                </Text>
                <BlockMath>{`d(A, B) = ||A - B|| = \\sqrt{\\sum_{i=1}^{n} (A_i - B_i)^2}`}</BlockMath>
              </Card>
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 4 }}>
              <Card shadow="xs" p="md" radius="md" withBorder>
                <Title order={4} mb="xs">Dot Product</Title>
                <Text size="sm" mb="sm">
                  Sum of the products of corresponding vector components, sensitive to both direction and magnitude.
                </Text>
                <BlockMath>{`A \\cdot B = \\sum_{i=1}^{n} A_i B_i`}</BlockMath>
              </Card>
            </Grid.Col>
          </Grid>
      </Card>

      <CodeBlock
        language="python"
        code={`
# Connecting to different vector databases with LangChain

# 1. Chroma
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_chroma = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# Query Chroma
docs = db_chroma.similarity_search("What is RAG?", k=3)

# 2. Pinecone
from langchain.vectorstores import Pinecone
import pinecone

# Initialize Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENV")
index_name = "langchain-demo"

# Create the index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384, metric="cosine")

# Load documents into Pinecone
db_pinecone = Pinecone.from_documents(documents, embeddings, index_name=index_name)

# 3. Weaviate
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

# Start embedded Weaviate (for local testing)
client = weaviate.Client(embedded_options=EmbeddedOptions())

# Create a Weaviate vector store
db_weaviate = Weaviate.from_documents(
    documents,
    embeddings,
    client=client,
    by_text=False,
    class_name="RAGDocuments"
)

# Advanced query with metadata filtering
query = "What is RAG?"
results = db_weaviate.similarity_search_with_score(
    query,
    k=3,
    where_filter={"source": "textbook"}  # Only retrieve from textbook sources
)
`}
      />

      <Space h="xl" />

      {/* 3. Document Chunking and Indexing */}
      <Title order={2} mb="sm" id="chunking">
        Document Chunking and Indexing
      </Title>
      <Text mb="md">
        Effective document processing is crucial for RAG system performance. This involves breaking down documents into
        appropriate chunks, embedding them, and indexing them for efficient retrieval.
      </Text>

      <Title order={3} mb="sm" id="document-chunking">
        Document Chunking Strategies
      </Title>
      <Text mb="md">
        Chunking involves splitting large documents into smaller, semantically meaningful units that can be efficiently
        embedded and retrieved. The choice of chunking strategy significantly impacts retrieval quality.
      </Text>

      <Accordion variant="separated" mb="lg">
        <Accordion.Item value="chunk-size">
          <Accordion.Control>
            <Text fw={500}>Chunk Size Considerations</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Text fw={500} mb="xs">Small Chunks (100-500 tokens)</Text>
                <List>
                  <List.Item><Text>More precise retrieval for specific information</Text></List.Item>
                  <List.Item><Text>Lower context utilization by LLM</Text></List.Item>
                  <List.Item><Text>Risk of losing broader context</Text></List.Item>
                  <List.Item><Text>Good for factoid QA and specific details</Text></List.Item>
                </List>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Text fw={500} mb="xs">Large Chunks (1000+ tokens)</Text>
                <List>
                  <List.Item><Text>Better preservation of context</Text></List.Item>
                  <List.Item><Text>More comprehensive information</Text></List.Item>
                  <List.Item><Text>Risk of diluting key information</Text></List.Item>
                  <List.Item><Text>Good for summarization and complex reasoning</Text></List.Item>
                </List>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="chunk-overlap">
          <Accordion.Control>
            <Text fw={500}>Chunk Overlap</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              Overlap between chunks helps maintain context continuity and prevents information from being lost at chunk boundaries.
              Typical overlap ranges from 10-20% of the chunk size.
            </Text>
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Text fw={500} mb="xs">Benefits of Overlap</Text>
                <List>
                  <List.Item><Text>Prevents loss of context at chunk boundaries</Text></List.Item>
                  <List.Item><Text>Improves retrieval of information that spans chunks</Text></List.Item>
                  <List.Item><Text>Provides redundancy for important information</Text></List.Item>
                </List>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Text fw={500} mb="xs">Considerations</Text>
                <List>
                  <List.Item><Text>Increases storage and computational requirements</Text></List.Item>
                  <List.Item><Text>Can lead to duplicate information in responses</Text></List.Item>
                  <List.Item><Text>Requires balancing with chunk size</Text></List.Item>
                </List>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="chunking-methods">
          <Accordion.Control>
            <Text fw={500}>Chunking Methods</Text>
          </Accordion.Control>
          <Accordion.Panel>
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
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <Title order={3} mb="sm" id="document-indexing">
        Document Indexing Process
      </Title>
      <Text mb="md">
        The indexing process involves generating embeddings for each document chunk and storing them in a vector database
        for efficient retrieval.
      </Text>

      <Grid mb="lg">
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Image
            src="/api/placeholder/640/480"
            alt="Document Processing Pipeline"
            caption="Document processing pipeline for RAG systems"
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
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
        </Grid.Col>
      </Grid>

      <CodeBlock
        language="python"
        code={`
# Advanced document chunking techniques
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain.document_loaders import PyPDFLoader

# Load a PDF file
loader = PyPDFLoader("technical_document.pdf")
pages = loader.load()

# 1. Character-based chunking (simple approach)
char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)
char_chunks = char_splitter.split_documents(pages)
print(f"Character-based chunks: {len(char_chunks)}")

# 2. Recursive character splitting (more sophisticated)
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", ", ", " "],  # Try splitting by these separators in order
    chunk_size=1000,
    chunk_overlap=200
)
recursive_chunks = recursive_splitter.split_documents(pages)
print(f"Recursive chunks: {len(recursive_chunks)}")

# 3. Token-based splitting (optimized for LLM token limits)
token_splitter = TokenTextSplitter(
    model_name="gpt-3.5-turbo",  # Use the same tokenizer as your LLM
    chunk_size=500,              # In tokens, not characters
    chunk_overlap=50
)
token_chunks = token_splitter.split_documents(pages)
print(f"Token-based chunks: {len(token_chunks)}")

# 4. Advanced token splitting for sentence-transformers
st_splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=256,  # Sentence transformer models often have smaller context windows
    chunk_overlap=25
)
st_chunks = st_splitter.split_documents(pages)
print(f"Sentence Transformers chunks: {len(st_chunks)}")

# Analyzing chunk quality
import numpy as np
from collections import Counter

# Calculate token distribution statistics
chunk_lengths = [len(chunk.page_content.split()) for chunk in recursive_chunks]
print(f"Average chunk size (words): {np.mean(chunk_lengths):.1f}")
print(f"Min chunk size: {min(chunk_lengths)}, Max chunk size: {max(chunk_lengths)}")
print(f"Standard deviation: {np.std(chunk_lengths):.1f}")

# Check for potential issues in chunks
problem_chunks = []
for i, chunk in enumerate(recursive_chunks):
    content = chunk.page_content
    if len(content) < 100:
        problem_chunks.append((i, "Too short", len(content)))
    elif not content.strip().endswith((".", "!", "?")):
        problem_chunks.append((i, "No sentence ending", content[-10:]))
    elif content.count("\n\n") > 5:
        problem_chunks.append((i, "Too many paragraph breaks", content.count("\n\n")))

print(f"Potential problem chunks: {len(problem_chunks)}")
for i, issue, detail in problem_chunks[:5]:  # Show first 5 issues
    print(f"Chunk {i}: {issue} - {detail}")
`}
      />

      <Space h="xl" />

      {/* 4. Similarity Search Algorithms */}
      <Title order={2} mb="sm" id="similarity-search">
        Similarity Search Algorithms
      </Title>
      <Text mb="md">
        Similarity search is the core retrieval operation in RAG systems. It involves finding the most relevant documents
        from a vector database given a query embedding.
      </Text>

        <Card shadow="sm" p="lg" radius="md" mb="lg" withBorder>
          <Title order={3} mb="sm">Approximate Nearest Neighbor (ANN) Search</Title>
          <Text mb="md">
            As vector databases grow to millions or billions of vectors, exact nearest neighbor search becomes computationally
            prohibitive. Approximate Nearest Neighbor (ANN) algorithms provide efficient alternatives that trade a small amount
            of accuracy for significant performance gains.
          </Text>

          <Grid mb="lg">
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Title order={4} mb="xs">Key ANN Algorithms</Title>
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
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Title order={4} mb="xs">Performance Metrics</Title>
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
                  <Text size="sm" mb="xs">
                    The number of queries that can be processed per second.
                  </Text>
                </List.Item>
                <List.Item mt="md">
                  <Text fw={500}>Index Build Time</Text>
                  <Text size="sm" mb="xs">
                    Time required to build the search index.
                  </Text>
                </List.Item>
                <List.Item mt="md">
                  <Text fw={500}>Memory Usage</Text>
                  <Text size="sm" mb="xs">
                    RAM required for index storage and search operations.
                  </Text>
                </List.Item>
              </List>
            </Grid.Col>
          </Grid>
        </Card>

      <Title order={3} mb="sm" id="hybrid-search">
        Hybrid Search Techniques
      </Title>
      <Text mb="md">
        Hybrid search combines multiple retrieval approaches to improve overall retrieval quality.
      </Text>

      <Grid mb="lg">
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" h="100%" withBorder>
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
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" h="100%" withBorder>
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
          </Card>
        </Grid.Col>
      </Grid>

      <CodeBlock
        language="python"
        code={`
# Advanced similarity search techniques with LangChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Base vector retriever
vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# 1. Using BM25 (keyword-based) retrieval alongside vector search
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# Create an ensemble retriever that combines both approaches
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # Equal weighting between keyword and semantic search
)

# Query the ensemble retriever
ensemble_results = ensemble_retriever.get_relevant_documents("What is transfer learning in NLP?")

# 2. Implementing a re-ranking approach
# First, get a larger candidate set
candidate_docs = vector_retriever.get_relevant_documents("What is transfer learning in NLP?")

# Define a re-ranking function using a cross-encoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load a cross-encoder model for relevance ranking
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def rerank_documents(query, docs, top_k=3):
    # Prepare inputs for cross-encoder
    pairs = [(query, doc.page_content) for doc in docs]
    
    # Tokenize and get model inputs
    features = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    
    # Get relevance scores
    with torch.no_grad():
        scores = model(**features).logits.flatten()
    
    # Sort documents by score
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k documents
    return [doc for doc, _ in doc_score_pairs[:top_k]]

# Get re-ranked documents
reranked_docs = rerank_documents("What is transfer learning in NLP?", candidate_docs)

# 3. Using LLM to extract relevant parts from retrieved documents
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={"temperature": 0.1}
)

# Create a compressor that extracts relevant information
compressor = LLMChainExtractor.from_llm(llm)

# Create a compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever
)

# Get compressed (extracted) documents
compressed_docs = compression_retriever.get_relevant_documents("What is transfer learning in NLP?")
print(f"Original document count: {len(candidate_docs)}")
print(f"Compressed document count: {len(compressed_docs)}")
print(f"Average original length: {sum(len(d.page_content) for d in candidate_docs) / len(candidate_docs)}")
print(f"Average compressed length: {sum(len(d.page_content) for d in compressed_docs) / len(compressed_docs)}")
`}
      />

      <Space h="xl" />

      {/* 5. Complete RAG Pipeline */}
      <Title order={2} mb="sm" id="pipeline">
        Complete RAG Pipeline
      </Title>
      <Text mb="md">
        Building a production-ready RAG system involves integrating all the previously discussed components into a coherent pipeline.
        This section covers the end-to-end implementation of a RAG system.
      </Text>

      <Title order={3} mb="sm" id="pipeline-components">
        Pipeline Components and Integration
      </Title>

      <Grid mb="lg">
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" withBorder>
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
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Image
            src="/api/placeholder/640/480"
            alt="RAG Pipeline Architecture"
            caption="End-to-end RAG pipeline architecture"
          />
        </Grid.Col>
      </Grid>

      <Title order={3} mb="sm" id="prompt-engineering">
        RAG Prompt Engineering
      </Title>
      <Text mb="md">
        The design of prompts for RAG systems is crucial for effectively leveraging the retrieved context and
        generating accurate responses.
      </Text>

      <Accordion variant="separated" mb="lg">
        <Accordion.Item value="prompt-components">
          <Accordion.Control>
            <Text fw={500}>Prompt Components</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
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
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card shadow="xs" p="md" radius="md" withBorder>
                  <Text fw={500} mb="xs">Example Prompt Template</Text>
                  <Code block>
                    {`You are a helpful AI assistant that answers questions based on the provided context.

CONTEXT:
{retrieved_documents}

USER QUESTION:
{query}

Answer the user's question using only the information provided in the context above. If the context doesn't contain the relevant information, say "I don't have enough information to answer this question." Do not make up information.

Provide your answer in a clear, concise manner. Include relevant quotes from the context where appropriate, marked with quotation marks.`}
                  </Code>
                </Card>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="prompt-strategies">
          <Accordion.Control>
            <Text fw={500}>Advanced Prompt Strategies</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card shadow="xs" p="md" radius="md" h="100%" withBorder>
                  <Title order={5} mb="xs">Chain-of-Verification</Title>
                  <Text size="sm">
                    Instructs the LLM to critically evaluate the retrieved information before generating a response.
                    The LLM first analyzes each retrieved passage for relevance and accuracy, then generates the final answer.
                  </Text>
                </Card>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card shadow="xs" p="md" radius="md" h="100%" withBorder>
                  <Title order={5} mb="xs">Step-by-Step Reasoning</Title>
                  <Text size="sm">
                    Explicitly guides the LLM to break down complex queries into sub-problems,
                    reason through each sub-problem using the retrieved context, and then synthesize a final answer.
                  </Text>
                </Card>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }} mt="md">
                <Card shadow="xs" p="md" radius="md" h="100%" withBorder>
                  <Title order={5} mb="xs">Source Attribution</Title>
                  <Text size="sm">
                    Requires the LLM to cite the specific documents or passages used for each part of the response,
                    improving transparency and enabling verification.
                  </Text>
                </Card>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }} mt="md">
                <Card shadow="xs" p="md" radius="md" h="100%" withBorder>
                  <Title order={5} mb="xs">Query Reformulation</Title>
                  <Text size="sm">
                    Instructs the LLM to rephrase or expand the original query to better match the knowledge base's content,
                    improving retrieval quality before answering.
                  </Text>
                </Card>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <CodeBlock
        language="python"
        code={`
# Building a complete RAG pipeline with LangChain

from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import logging
import os
from datetime import datetime

# 1. Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. Configure pipeline parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 5

# 3. Document Loading
def load_documents(sources):
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
            logger.info(f"Loaded {len(documents)} documents from {source}")
        except Exception as e:
            logger.error(f"Error loading {source}: {str(e)}")
    
    return documents

# 4. Document Processing
def process_documents(documents):
    """Split documents into chunks and generate embeddings"""
    logger.info("Processing documents...")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Add metadata for tracking
    for i, chunk in enumerate(chunks):
        if 'source' not in chunk.metadata:
            chunk.metadata['source'] = f"document_{i}"
        chunk.metadata['chunk_id'] = i
        chunk.metadata['timestamp'] = datetime.now().isoformat()
    
    # Create embeddings and vector store
    logger.info(f"Generating embeddings using {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    logger.info(f"Vector database created with {len(chunks)} entries")
    
    return vectordb

# 5. RAG Chain Setup
def setup_rag_chain(vectordb):
    """Configure the RAG retrieval and generation chain"""
    logger.info("Setting up RAG chain...")
    
    # Configure retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})
    
    # Set up LLM
    logger.info(f"Initializing LLM: {LLM_MODEL}")
    llm = HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL,
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 512},
        device=0  # Use GPU if available
    )
    
    # Create prompt template with context
    template = """
    You are a helpful AI assistant that answers questions based on the provided context.
    
    CONTEXT:
    {context}
    
    USER QUESTION:
    {question}
    
    Answer the user's question using only the information provided in the context above. 
    If the context doesn't contain the relevant information, say "I don't have enough information to answer this question." 
    Do not make up information.
    
    Provide your answer in a clear, concise manner. Include relevant quotes from the context where appropriate, marked with quotation marks.
    Also provide the source of the information if available in the metadata.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create the RAG chain
    chain_type_kwargs = {"prompt": prompt}
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Other options: "map_reduce", "refine", "map_rerank"
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,  # Include source docs in the response
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    logger.info("RAG chain setup complete")
    return rag_chain

# 6. Query Execution
def execute_query(rag_chain, query):
    """Execute a query against the RAG system"""
    logger.info(f"Executing query: {query}")
    
    try:
        # Process the query
        start_time = datetime.now()
        result = rag_chain({"query": query})
        end_time = datetime.now()
        
        # Log performance metrics
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        # Return results
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return {"error": str(e)}

# 7. Main RAG Pipeline
def run_rag_pipeline(sources, query):
    """Run the complete RAG pipeline"""
    # Load documents
    documents = load_documents(sources)
    
    # Process documents
    vectordb = process_documents(documents)
    
    # Setup RAG chain
    rag_chain = setup_rag_chain(vectordb)
    
    # Execute query
    result = execute_query(rag_chain, query)
    
    return result

# Example usage
if __name__ == "__main__":
    sources = [
        "data/nlp_textbook.pdf",
        "data/transformer_paper.txt",
        "https://en.wikipedia.org/wiki/Natural_language_processing"
    ]
    
    query = "Explain how transformers revolutionized NLP tasks"
    
    result = run_rag_pipeline(sources, query)
    
    # Print the answer
    print("\nFinal Answer:")
    print(result["answer"])
    
    # Print source information
    print("\nSources:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}")
        
    # Print performance metrics
    print(f"\nProcessing time: {result['processing_time']:.2f} seconds")
`}
      />

      <Space h="xl" />

      {/* 6. RAG System Evaluation */}
      <Title order={2} mb="sm" id="evaluation">
        RAG System Evaluation
      </Title>
      <Text mb="md">
        Evaluating RAG systems is essential for measuring performance, identifying bottlenecks, and ensuring
        the system meets quality requirements. This involves assessing both retrieval and generation components.
      </Text>

      <Grid mb="lg">
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" h="100%" withBorder>
            <Title order={3} mb="sm">Retrieval Evaluation Metrics</Title>
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
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" h="100%" withBorder>
            <Title order={3} mb="sm">Generation Evaluation Metrics</Title>
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
          </Card>
        </Grid.Col>
      </Grid>

      <Title order={3} mb="sm" id="evaluation-methods">
        Evaluation Methods
      </Title>

      <Accordion variant="separated" mb="lg">
        <Accordion.Item value="manual">
          <Accordion.Control>
            <Text fw={500}>Manual Evaluation</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              Human evaluation provides in-depth assessment of RAG system outputs but is resource-intensive and difficult to scale.
            </Text>
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Text fw={500} mb="xs">Approaches</Text>
                <List>
                  <List.Item><Text>Expert review of response accuracy</Text></List.Item>
                  <List.Item><Text>Side-by-side comparison with baseline systems</Text></List.Item>
                  <List.Item><Text>Human rating on multiple quality dimensions</Text></List.Item>
                  <List.Item><Text>User satisfaction surveys</Text></List.Item>
                </List>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Text fw={500} mb="xs">Best Practices</Text>
                <List>
                  <List.Item><Text>Use standardized evaluation rubrics</Text></List.Item>
                  <List.Item><Text>Ensure diverse evaluator backgrounds</Text></List.Item>
                  <List.Item><Text>Implement inter-rater reliability checks</Text></List.Item>
                  <List.Item><Text>Balance qualitative and quantitative feedback</Text></List.Item>
                </List>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="automatic">
          <Accordion.Control>
            <Text fw={500}>Automatic Evaluation</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Text mb="md">
              Automatic evaluation methods enable large-scale assessment of RAG systems and provide consistent metrics for comparison.
            </Text>
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Text fw={500} mb="xs">Reference-Based Methods</Text>
                <List>
                  <List.Item><Text>ROUGE (Recall-Oriented Understudy for Gisting Evaluation)</Text></List.Item>
                  <List.Item><Text>BLEU (Bilingual Evaluation Understudy)</Text></List.Item>
                  <List.Item><Text>METEOR (Metric for Evaluation of Translation with Explicit Ordering)</Text></List.Item>
                  <List.Item><Text>BERTScore (contextual embedding similarity)</Text></List.Item>
                </List>
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Text fw={500} mb="xs">LLM-as-a-Judge Methods</Text>
                <List>
                  <List.Item><Text>LLM-based factuality checking</Text></List.Item>
                  <List.Item><Text>Consistency evaluation with retrieved context</Text></List.Item>
                  <List.Item><Text>Multi-dimensional quality assessment</Text></List.Item>
                  <List.Item><Text>Question answering accuracy evaluation</Text></List.Item>
                </List>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="benchmark">
          <Accordion.Control>
            <Text fw={500}>Benchmarking Datasets</Text>
          </Accordion.Control>
          <Accordion.Panel>
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
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <CodeBlock
        language="python"
        code={`
# RAG system evaluation with LangChain
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain.evaluation.schema import StringEvaluator
from langchain.evaluation.criteria import LabeledCriteriaEvaluator
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load a question-answering dataset for evaluation
dataset = load_dataset("nq_open", split="validation[:100]")  # Load first 100 examples

# 1. Retrieval Evaluation Functions
def evaluate_retrieval(retriever, dataset, ground_truth_docs):
    """Evaluate retriever performance"""
    results = {
        "precision": [],
        "recall": [],
        "reciprocal_rank": [],
        "query_time": []
    }
    
    for example in tqdm(dataset):
        query = example["question"]
        relevant_doc_ids = ground_truth_docs.get(query, [])
        
        # Time the retrieval
        start_time = datetime.now()
        retrieved_docs = retriever.get_relevant_documents(query)
        query_time = (datetime.now() - start_time).total_seconds()
        
        # Get IDs of retrieved docs
        retrieved_ids = [doc.metadata.get("id") for doc in retrieved_docs]
        
        # Calculate metrics
        relevant_retrieved = set(relevant_doc_ids).intersection(set(retrieved_ids))
        
        # Precision@k where k is the number of retrieved docs
        precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
        
        # Recall
        recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0
        
        # Reciprocal Rank
        rr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_doc_ids:
                rr = 1 / (i + 1)
                break
        
        # Store results
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["reciprocal_rank"].append(rr)
        results["query_time"].append(query_time)
    
    # Calculate averages
    avg_results = {
        "avg_precision": np.mean(results["precision"]),
        "avg_recall": np.mean(results["recall"]),
        "mean_reciprocal_rank": np.mean(results["reciprocal_rank"]),
        "avg_query_time": np.mean(results["query_time"])
    }
    
    return avg_results, results

# 2. LLM-based Response Evaluation
def evaluate_generation(rag_chain, dataset, criteria=None):
    """Evaluate the quality of generated responses"""
    if criteria is None:
        criteria = {
            "faithfulness": "Does the response only contain information that is present in the retrieved context?",
            "relevance": "Does the response directly address the user's question?",
            "coherence": "Is the response well-structured, logical, and easy to follow?",
            "conciseness": "Is the response concise and to the point without unnecessary information?"
        }
    
    # Initialize LLM-based evaluators
    faithfulness_evaluator = load_evaluator(
        EvaluatorType.LABELED_CRITERIA,
        criteria={"faithfulness": criteria["faithfulness"]},
        normalize_by="length"
    )
    
    qa_evaluator = load_evaluator(EvaluatorType.QA)
    
    # Run evaluation
    results = []
    for example in tqdm(dataset):
        query = example["question"]
        reference_answer = example["answer"]
        
        # Get RAG response
        response = rag_chain({"query": query})
        generated_answer = response["result"]
        context = "\n".join([doc.page_content for doc in response["source_documents"]])
        
        # Evaluate faithfulness
        faithfulness_result = faithfulness_evaluator.evaluate_strings(
            prediction=generated_answer,
            reference=context,
            input=query
        )
        
        # Evaluate QA accuracy
        qa_result = qa_evaluator.evaluate_strings(
            prediction=generated_answer,
            reference=reference_answer,
            input=query
        )
        
        # Store results
        results.append({
            "query": query,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "faithfulness_score": faithfulness_result["normalized_score"],
            "qa_score": qa_result["score"],
            "context": context
        })
    
    # Calculate aggregate metrics
    df = pd.DataFrame(results)
    
    metrics = {
        "avg_faithfulness": df["faithfulness_score"].mean(),
        "avg_qa_score": df["qa_score"].mean(),
        "perfect_qa_percentage": (df["qa_score"] == 1.0).mean() * 100
    }
    
    return metrics, df

# 3. RAG Tracing and Debugging
def analyze_failure_cases(evaluation_results, threshold=0.5):
    """Analyze cases where the RAG system performed poorly"""
    df = pd.DataFrame(evaluation_results)
    
    # Identify low-performing cases
    low_faithfulness = df[df["faithfulness_score"] < threshold]
    low_qa_score = df[df["qa_score"] < threshold]
    
    # Analyze retrieval issues
    retrieval_issues = []
    for _, row in low_qa_score.iterrows():
        query = row["query"]
        context = row["context"]
        
        # Check if context contains relevant information
        if len(context) < 100 or row["faithfulness_score"] > 0.8:
            retrieval_issues.append({
                "query": query,
                "issue": "Insufficient relevant context retrieved",
                "context_length": len(context)
            })
    
    # Analyze generation issues
    generation_issues = []
    for _, row in low_faithfulness.iterrows():
        if row["qa_score"] > threshold:  # Correct answer but unfaithful
            generation_issues.append({
                "query": row["query"],
                "issue": "Hallucination - correct answer without context support",
                "generated_answer": row["generated_answer"],
                "context": row["context"][:200] + "..."  # Truncate for readability
            })
    
    return {
        "retrieval_issues": retrieval_issues,
        "generation_issues": generation_issues,
        "low_faithfulness_count": len(low_faithfulness),
        "low_qa_score_count": len(low_qa_score)
    }

# 4. End-to-End System Evaluation
def evaluate_rag_system(sources, test_dataset, ground_truth_docs=None):
    """Run comprehensive evaluation of the RAG system"""
    # Load documents and prepare system
    documents = load_documents(sources)
    vectordb = process_documents(documents)
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})
    rag_chain = setup_rag_chain(vectordb)
    
    # Evaluate retrieval component (if ground truth is available)
    retrieval_metrics = None
    if ground_truth_docs:
        retrieval_metrics, retrieval_details = evaluate_retrieval(
            retriever, test_dataset, ground_truth_docs
        )
        print("\nRetrieval Metrics:")
        for metric, value in retrieval_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Evaluate generation component
    generation_metrics, generation_details = evaluate_generation(
        rag_chain, test_dataset
    )
    
    print("\nGeneration Metrics:")
    for metric, value in generation_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Analyze failure cases
    if len(generation_details) > 0:
        failure_analysis = analyze_failure_cases(generation_details)
        print(f"\nIdentified {failure_analysis['low_faithfulness_count']} faithfulness issues")
        print(f"Identified {failure_analysis['low_qa_score_count']} answer quality issues")
    
    # Return all metrics and details
    return {
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "retrieval_details": retrieval_details if ground_truth_docs else None,
        "generation_details": generation_details,
        "failure_analysis": failure_analysis if len(generation_details) > 0 else None
    }

# Example usage
if __name__ == "__main__":
    # Define test dataset and sources
    test_dataset = load_dataset("squad", split="validation[:50]")
    
    sources = [
        "data/nlp_textbook.pdf",
        "data/research_papers/",
        "https://en.wikipedia.org/wiki/Natural_language_processing"
    ]
    
    # Run evaluation
    eval_results = evaluate_rag_system(sources, test_dataset)
    
    # Export detailed results to CSV for further analysis
    if eval_results["generation_details"] is not None:
        eval_results["generation_details"].to_csv("rag_evaluation_results.csv", index=False)
`}
      />

      <Space h="xl" />

      {/* 7. Prompt Techniques */}
      <Title order={2} mb="sm" id="prompt-techniques">
        Prompt Techniques
      </Title>
      <Text mb="md">
        Effective prompt design is crucial for maximizing the performance of RAG systems. This section covers
        advanced prompt techniques specifically tailored for retrieval-augmented generation.
      </Text>

      <Title order={3} mb="sm" id="retrieval-focused">
        Retrieval-Focused Prompt Techniques
      </Title>

      <Grid mb="lg">
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" withBorder>
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
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" withBorder>
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
          </Card>
        </Grid.Col>
      </Grid>

      <Title order={3} mb="sm" id="generation-focused">
        Generation-Focused Prompt Techniques
      </Title>

      <Grid mb="lg">
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" withBorder>
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
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card shadow="sm" p="lg" radius="md" withBorder>
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
          </Card>
        </Grid.Col>
      </Grid>

      <Card shadow="sm" p="lg" radius="md" mb="lg" withBorder>
        <Title order={3} mb="sm">Advanced RAG Prompt Template</Title>
        <Text mb="md">
          This template incorporates multiple advanced techniques for high-quality RAG responses.
        </Text>
        <Code block>
{`You are an AI assistant that provides accurate, helpful responses based on the provided context.

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
   - Include direct quotes from the context where appropriate, using quotation marks
   - Cite the specific source (document title or ID) for each key piece of information

3) VERIFICATION:
   - Review your response to ensure all claims are supported by the context
   - Remove any statements not directly supported by the provided information
   - Assess your confidence in the accuracy of the response

Your response should be:
- Comprehensive yet concise
- Directly relevant to the query
- Well-structured with logical flow
- Factually accurate and supported by the context
- Free from speculation or information not in the context

Begin your response now:`}
        </Code>
      </Card>

      <CodeBlock
        language="python"
        code={`
# Advanced RAG prompt techniques with LangChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder

# 1. Hypothetical Document Embeddings (HyDE)
def setup_hyde_retriever(vectordb, llm):
    """Create a HyDE retriever that generates a hypothetical document before retrieval"""
    
    # Create the prompt for generating hypothetical documents
    hyde_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an expert on the topic of NLP and language models.
        
Write a passage that directly answers this question:
{question}
        
Make the passage detailed, informative, and factually correct. 
Include technical details where appropriate."""
    )
    
    # Create the HyDE embedder
    hyde_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=LLMChain(llm=llm, prompt=hyde_prompt),
        base_embeddings=vectordb.embeddings
    )
    
    # Create a new vectorstore with the HyDE embeddings
    hyde_vectordb = vectordb.as_retriever(
        search_kwargs={"k": 5},
        search_type="similarity",
    )
    
    return hyde_vectordb

# 2. Self-Query Retriever
def setup_self_query_retriever(vectordb, llm):
    """Create a retriever that constructs its own filters based on the query"""
    
    # Define metadata fields and descriptions
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The source document or URL where this information comes from",
            type="string",
        ),
        AttributeInfo(
            name="date",
            description="The date the document was published or last updated",
            type="string",
        ),
        AttributeInfo(
            name="author",
            description="The author of the document",
            type="string",
        ),
        AttributeInfo(
            name="topic",
            description="The main topic or category of the document (e.g., 'transformers', 'embeddings', 'evaluation')",
            type="string",
        )
    ]
    
    # Create the document content description
    document_content_description = "Academic and technical information about natural language processing, embeddings, transformers, and language models"
    
    # Create the self-query retriever
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectordb,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True
    )
    
    return retriever

# 3. Query Decomposition
def decompose_query(llm, query):
    """Break down a complex query into simpler sub-queries"""
    
    decomposition_prompt = PromptTemplate(
        input_variables=["query"],
        template="""You are an expert at breaking down complex questions into simpler sub-questions.

Original Question: {query}

Break this question down into 2-4 simpler sub-questions that would help answer the original question when combined.
Each sub-question should be focused on a specific aspect and be answerable independently.

Output the sub-questions as a numbered list without any additional text."""
    )
    
    decomposition_chain = LLMChain(llm=llm, prompt=decomposition_prompt)
    result = decomposition_chain.run(query=query)
    
    # Parse sub-queries (assuming they come as a numbered list)
    sub_queries = []
    for line in result.strip().split("\n"):
        line = line.strip()
        if line and (line.startswith("- ") or line.startswith("* ") or 
                    any(line.startswith(f"{i}.") for i in range(1, 10))):
            # Remove numbering/bullets and strip whitespace
            clean_line = line.split(".", 1)[-1].strip() if "." in line else line[2:].strip()
            sub_queries.append(clean_line)
    
    return sub_queries

# 4. Prompt with Chain-of-Verification
def create_verification_chain(llm, retriever):
    """Create a chain that verifies information before generating a response"""
    
    verification_prompt = PromptTemplate(
        template="""You are a careful and precise assistant that answers questions based on the retrieved context.

CONTEXT:
{context}

USER QUESTION:
{question}

Please follow these steps:

1. VERIFICATION PHASE:
   - Analyze each piece of retrieved information for relevance to the question
   - Check for any contradictions or inconsistencies in the retrieved context
   - Evaluate the reliability and completeness of the information

2. THOUGHT PROCESS:
   - Based on the verified information, outline your reasoning process
   - Consider different perspectives or interpretations
   - Identify any remaining uncertainties or gaps in information

3. ANSWER FORMULATION:
   - Provide a direct answer to the user's question
   - Only include information that is explicitly supported by the context
   - If the context doesn't contain enough information, acknowledge this limitation
   - Cite specific parts of the context to support your answer

VERIFICATION ANALYSIS:

THOUGHT PROCESS:

ANSWER:""",
        input_variables=["context", "question"]
    )
    
    # Create retrieval chain with source attribution
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": verification_prompt},
        reduce_k_below_max_tokens=True,
        max_tokens_limit=3800  # Adjust based on your LLM's context window
    )
    
    return chain

# 5. Multi-step RAG with FLARE-like approach
def flare_rag_generation(llm, retriever, query, max_iterations=3):
    """
    Implement a FLARE-like approach where generation is interleaved with retrieval
    FLARE: Forward-Looking Active REtrieval
    """
    # Initial retrieval
    context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(query)])
    
    # Initial prompt
    system_prompt = """You are an AI assistant that answers questions based on retrieved context.
As you generate your response, identify when you need more information and specify what to search for.
When you need more information, write [SEARCH: your specific search query] on a new line.
After all necessary searches, provide your final answer that addresses the original question."""
    
    # Initialize response generation
    partial_response = ""
    search_results = [context]
    
    for i in range(max_iterations):
        # Construct current prompt with all context and partial response
        current_prompt = f"{system_prompt}\n\nCONTEXT:\n"
        for idx, ctx in enumerate(search_results):
            current_prompt += f"--- Context {idx+1} ---\n{ctx}\n\n"
        
        current_prompt += f"QUESTION: {query}\n\nRESPONSE SO FAR:\n{partial_response}\n\nContinue generating your response:"
        
        # Get next part of response
        next_part = llm(current_prompt)
        
        # Check if a search is requested
        if "[SEARCH:" in next_part:
            # Split at the search request
            before_search, after_search = next_part.split("[SEARCH:", 1)
            search_query, remaining = after_search.split("]", 1) if "]" in after_search else (after_search, "")
            
            # Add the text before the search to partial response
            partial_response += before_search
            
            # Perform the additional retrieval
            new_docs = retriever.get_relevant_documents(search_query.strip())
            new_context = "\n\n".join([doc.page_content for doc in new_docs])
            search_results.append(f"Results for query '{search_query.strip()}':\n{new_context}")
            
            # Continue with remaining text
            partial_response += remaining
        else:
            # No more searches needed, add the full response
            partial_response += next_part
            break
    
    return partial_response

# Example usage of these techniques
def advanced_rag_example():
    # Initialize base components
    documents = load_documents(["knowledge_base/"])
    vectordb = process_documents(documents)
    llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large", task="text2text-generation")
    
    # Example 1: HyDE retriever
    hyde_retriever = setup_hyde_retriever(vectordb, llm)
    
    # Example 2: Self-query retriever
    self_query_retriever = setup_self_query_retriever(vectordb, llm)
    
    # Example 3: Query decomposition
    complex_query = "How do transformer models improve on RNNs and what are the key innovations that make BERT different from previous approaches?"
    sub_queries = decompose_query(llm, complex_query)
    
    # Example 4: Chain-of-verification
    verification_chain = create_verification_chain(llm, vectordb.as_retriever())
    
    # Example 5: FLARE approach
    flare_response = flare_rag_generation(llm, vectordb.as_retriever(), complex_query)
    
    # Return sample outputs
    return {
        "hyde_docs": hyde_retriever.get_relevant_documents("How do transformers handle sequential data?"),
        "self_query_result": self_query_retriever.get_relevant_documents("Find recent papers about BERT from 2023"),
        "decomposed_queries": sub_queries,
        "verification_result": verification_chain({"question": "What are the advantages of transformer models?"}),
        "flare_result": flare_response
    }
`}
      />
      <Space h="xl" />

      <Title order={2} mb="sm" id="further-reading">
        Further Reading
      </Title>
      <List>
        <List.Item>
          <Text fw={500}>Research Papers</Text>
          <List withPadding>
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
        </List.Item>
        <List.Item mt="md">
          <Text fw={500}>Tutorials and Resources</Text>
          <List withPadding>
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
        </List.Item>
      </List>

      <Space h="xl" />

      <Card shadow="sm" p="lg" radius="md" mb="xl" withBorder>
        <Title order={3} mb="sm">Key Takeaways</Title>
        <Grid>
          <Grid.Col span={{ base: 12, md: 6 }}>
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
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
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
          </Grid.Col>
        </Grid>
      </Card>
    </Container>
  );
};

export default RAG;