import React from 'react';
import { Container, Title, Text, Accordion, Table, Stack, Paper, List } from '@mantine/core';
import { Database, Table as TableIcon } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const DatabaseIcon = ({ type }) => {
  const iconProps = { size: 24, strokeWidth: 1.5 };
  return type === 'sql' ? <TableIcon {...iconProps} /> : <Database {...iconProps} />;
};

const DatabaseItem = ({ type, name, description, characteristics, examples, code }) => (
  <Accordion.Item value={type}>
    <Accordion.Control icon={<DatabaseIcon type={type} />}>
      {name}
    </Accordion.Control>
    <Accordion.Panel>
      <Stack spacing="md">
        <Text>{description}</Text>
        <Title order={4}>Characteristics</Title>
        <List>
          {characteristics.map((char, index) => (
            <List.Item key={index}>{char}</List.Item>
          ))}
        </List>
        <Title order={4}>Examples</Title>
        <List>
          {examples.map((example, index) => (
            <List.Item key={index}>{example}</List.Item>
          ))}
        </List>
        <Title order={4}>Interacting with the Database</Title>
        <CodeBlock language="python" code={code} />
      </Stack>
    </Accordion.Panel>
  </Accordion.Item>
);

const Databases = () => {
  const databaseTypes = [
    {
      type: 'sql',
      name: 'SQL Databases',
      description: 'SQL (Structured Query Language) databases are relational databases that use structured query language for defining and manipulating the data.',
      characteristics: [
        'Table-based structure with predefined schema',
        'Support for complex queries and joins',
        'ACID (Atomicity, Consistency, Isolation, Durability) compliance',
        'Vertical scalability (scaling up)',
        'Best for complex queries and transactions',
      ],
      examples: [
        'PostgreSQL',
        'MySQL',
        'Oracle',
        'Microsoft SQL Server',
        'SQLite',
      ],
      code: `import psycopg2

# Connect to the database
conn = psycopg2.connect(
    dbname="your_database",
    user="your_username",
    password="your_password",
    host="your_host",
    port="your_port"
)

# Create a cursor
cur = conn.cursor()

# Execute a query
cur.execute("SELECT * FROM users WHERE age > 25")

# Fetch the results
results = cur.fetchall()

# Print the results
for row in results:
    print(row)

# Close the cursor and connection
cur.close()
conn.close()

# Using SQLAlchemy ORM
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

engine = create_engine('postgresql://username:password@host:port/dbname')
Session = sessionmaker(bind=engine)
session = Session()

# Query using ORM
users = session.query(User).filter(User.age > 25).all()

for user in users:
    print(f"{user.name}, {user.age}")

session.close()`
    },
    {
      type: 'nosql',
      name: 'NoSQL Databases',
      description: 'NoSQL (Not Only SQL) databases provide a mechanism for storage and retrieval of data that is modeled in means other than the tabular relations used in relational databases.',
      characteristics: [
        'Flexible schema or schema-less',
        'Horizontal scalability (scaling out)',
        'Eventual consistency (in some cases)',
        'Optimized for specific data models and access patterns',
        'Best for handling large volumes of unstructured or semi-structured data',
      ],
      examples: [
        'MongoDB (Document-based)',
        'Cassandra (Column-family)',
        'Redis (Key-value)',
        'Neo4j (Graph)',
        'Elasticsearch (Search engine)',
      ],
      code: `# Using MongoDB with pymongo
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database']
collection = db['your_collection']

# Insert a document
document = {"name": "John Doe", "age": 30, "city": "New York"}
result = collection.insert_one(document)
print(f"Inserted document ID: {result.inserted_id}")

# Query documents
query = {"age": {"$gt": 25}}
results = collection.find(query)

for doc in results:
    print(doc)

# Update a document
update_query = {"name": "John Doe"}
new_values = {"$set": {"age": 31}}
collection.update_one(update_query, new_values)

# Delete a document
delete_query = {"name": "John Doe"}
collection.delete_one(delete_query)

client.close()

# Using Redis with redis-py
import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Set a key-value pair
r.set('user:1', 'John Doe')

# Get a value
value = r.get('user:1')
print(value)

# Set with expiration
r.setex('session:42', 3600, 'active')  # expires in 1 hour

# Increment a counter
r.incr('pageviews')

# Delete a key
r.delete('user:1')

r.close()`
    }
  ];

  return (
    <Container fluid>
      <Title order={1}>Databases</Title>
      <Text mt="md">
        Databases are essential components in data science workflows, providing structured storage and efficient retrieval of large amounts of data. In this section, we'll explore the two main categories of databases: SQL (relational) and NoSQL (non-relational) databases.
      </Text>
      
      <Title order={2} mt="xl">Database Comparison</Title>
      <Table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>SQL Databases</th>
            <th>NoSQL Databases</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Data Model</td>
            <td>Relational (tables)</td>
            <td>Various (document, key-value, column-family, graph)</td>
          </tr>
          <tr>
            <td>Schema</td>
            <td>Fixed, predefined</td>
            <td>Flexible or schema-less</td>
          </tr>
          <tr>
            <td>Scalability</td>
            <td>Vertical (scale up)</td>
            <td>Horizontal (scale out)</td>
          </tr>
          <tr>
            <td>ACID Compliance</td>
            <td>Yes</td>
            <td>Varies (some offer eventual consistency)</td>
          </tr>
          <tr>
            <td>Query Language</td>
            <td>SQL (standardized)</td>
            <td>Database-specific</td>
          </tr>
          <tr>
            <td>Best For</td>
            <td>Complex queries, transactions</td>
            <td>Large volumes of unstructured data, rapid development</td>
          </tr>
        </tbody>
      </Table>

      <Accordion mt="xl">
        {databaseTypes.map(db => (
          <DatabaseItem key={db.type} {...db} />
        ))}
      </Accordion>
    </Container>
  );
};

export default Databases;