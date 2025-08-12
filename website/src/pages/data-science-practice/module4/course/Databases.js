import React from 'react';
import { Container, Title, Text, Accordion, Stack, List, Paper, Table } from '@mantine/core';
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
      <Stack gap="md">
        <Text>{description}</Text>
        <Title order={4}>Characteristics</Title>
        <List>
          {characteristics.map((char, index) => (
            <List.Item key={index}>{char}</List.Item>
          ))}
        </List>
        <Title order={4}>Popular Examples</Title>
        <List>
          {examples.map((example, index) => (
            <List.Item key={index}>{example}</List.Item>
          ))}
        </List>
        <Title order={4}>Basic Usage Principles</Title>
        <CodeBlock language="python" code={code} />
      </Stack>
    </Accordion.Panel>
  </Accordion.Item>
);

const Databases = () => {
  // Enhanced comparison data structure
  const comparisonData = [
    {
      feature: "Data Model",
      sql: "Tabular with fixed rows and columns",
      nosql: "Flexible: Documents, Key-Value, Graph, etc.",
      description: "How data is organized and structured"
    },
    {
      feature: "Schema",
      sql: "Strict, predefined schema required",
      nosql: "Dynamic, schema-less or flexible schema",
      description: "Rules that define data organization"
    },
    {
      feature: "Relationships",
      sql: "Built-in support via foreign keys and JOINs",
      nosql: "Handled at application level or denormalized",
      description: "How data connections are managed"
    },
    {
      feature: "Scalability",
      sql: "Vertical (better hardware)",
      nosql: "Horizontal (more servers)",
      description: "How system handles increased load"
    },
    {
      feature: "Consistency",
      sql: "ACID guarantees",
      nosql: "Eventually consistent (BASE)",
      description: "Data reliability and transaction handling"
    },
    {
      feature: "Use Cases",
      sql: "Complex queries, transactions",
      nosql: "High throughput, flexible data, rapid changes",
      description: "Ideal application scenarios"
    }
  ];

  const databaseTypes = [
    {
      type: 'sql',
      name: 'SQL Databases',
      description: 'Relational databases using structured tables with predefined schemas and relationships.',
      characteristics: [
        'Fixed schema with tables, rows, and columns',
        'Strong data consistency and ACID compliance',
        'Powerful querying with JOIN operations',
        'Best for complex relationships and transactions',
        'Mature ecosystem with standardized language (SQL)',
      ],
      examples: [
        'PostgreSQL - Advanced open-source RDBMS',
        'MySQL - Popular open-source database',
        'SQLite - Lightweight, serverless database',
      ],
      code: `# Basic SQL Database Operations Example
from sqlalchemy import create_engine, text

# Create database connection
engine = create_engine('database_url')

# Basic CRUD Operations
with engine.connect() as conn:
    # Create - Insert data
    conn.execute(text("""
        INSERT INTO users (name, email) 
        VALUES (:name, :email)
    """), {"name": "John", "email": "john@example.com"})
    
    # Read - Query data
    result = conn.execute(text("""
        SELECT * FROM users 
        WHERE email LIKE :pattern
    """), {"pattern": "%@example.com"})
    
    # Update - Modify data
    conn.execute(text("""
        UPDATE users 
        SET status = :status 
        WHERE email = :email
    """), {"status": "active", "email": "john@example.com"})
    
    # Delete - Remove data
    conn.execute(text("""
        DELETE FROM users 
        WHERE email = :email
    """), {"email": "john@example.com"})

# Transaction Example
with engine.begin() as conn:
    try:
        conn.execute(text("UPDATE accounts SET balance = balance - 100 WHERE id = 1"))
        conn.execute(text("UPDATE accounts SET balance = balance + 100 WHERE id = 2"))
    except Exception as e:
        # Transaction automatically rolls back on error
        print(f"Transfer failed: {e}")
        raise`
    },
    {
      type: 'nosql',
      name: 'NoSQL Databases',
      description: 'Non-relational databases optimized for flexible data models and horizontal scaling.',
      characteristics: [
        'Flexible schema for evolving data structures',
        'Horizontal scalability for large datasets',
        'Optimized for specific data models and patterns',
        'Eventually consistent in distributed scenarios',
        'High performance for specific use cases',
      ],
      examples: [
        'MongoDB - Document store for JSON-like data',
        'Redis - In-memory key-value store',
        'Neo4j - Graph database for connected data',
      ],
      code: `# NoSQL Database Operations Example (MongoDB-style)
from pymongo import MongoClient

# Connect to database
client = MongoClient('database_url')
db = client.database_name
collection = db.collection_name

# Basic CRUD Operations

# Create - Insert documents
doc = {
    "name": "John",
    "email": "john@example.com",
    "preferences": {
        "theme": "dark",
        "notifications": True
    }
}
result = collection.insert_one(doc)

# Read - Query documents
# Simple query
user = collection.find_one({"email": "john@example.com"})

# Complex query
users = collection.find({
    "preferences.theme": "dark",
    "age": {"$gt": 25}
})

# Update - Modify documents
collection.update_one(
    {"email": "john@example.com"},
    {
        "$set": {"status": "active"},
        "$push": {"login_times": "2024-01-01"}
    }
)

# Delete - Remove documents
collection.delete_one({"email": "john@example.com"})

# Aggregation Example
pipeline = [
    {"$match": {"status": "active"}},
    {"$group": {
        "_id": "$country",
        "user_count": {"$sum": 1}
    }}
]
results = collection.aggregate(pipeline)`
    }
  ];

  return (
    <Container fluid>
      <Stack gap="xl">
        <div>
          <Title order={1}>Databases</Title>
          <Text mt="md">
            Databases are fundamental to data science, providing organized storage and efficient retrieval of data. 
            Understanding the differences between SQL and NoSQL databases is crucial for choosing the right tool for your data needs.
          </Text>
        </div>

        <div>
          <Title order={2} id="comparison">Database Comparison</Title>
          <Paper p="md" radius="md" className="bg-slate-50">
            <Table striped highlightOnHover>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>SQL Databases</th>
                  <th>NoSQL Databases</th>
                </tr>
              </thead>
              <tbody>
                {comparisonData.map((row, index) => (
                  <tr key={index}>
                    <td><Text fw={500}>{row.feature}</Text></td>
                    <td>{row.sql}</td>
                    <td>{row.nosql}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Paper>
        </div>

        <div>
          <Title order={2} id="implementations">Database Implementations</Title>
          <Accordion>
            {databaseTypes.map(db => (
              <DatabaseItem key={db.type} {...db} />
            ))}
          </Accordion>
        </div>
      </Stack>
    </Container>
  );
};

export default Databases;