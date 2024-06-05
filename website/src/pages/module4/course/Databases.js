import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Databases = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Databases</h1>
      <p>
        In this section, you will learn how to retrieve data from databases
        using Python.
      </p>
      <Row>
        <Col>
          <h2>SQL Databases</h2>
          <p>
            SQL (Structured Query Language) databases are a common format for
            storing structured data. To connect to a SQL database in Python, you
            can use the `sqlite3` or `psycopg2` library.
          </p>
          <CodeBlock
            code={`import sqlite3

conn = sqlite3.connect("data.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM data")
data = cursor.fetchall()`}
          />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>NoSQL Databases</h2>
          <p>
            NoSQL (Not Only SQL) databases are a non-relational format for
            storing unstructured data. To connect to a NoSQL database in Python,
            you can use the `pymongo` library.
          </p>
          <CodeBlock
            code={`from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["database"]
collection = db["data"]

data = collection.find()`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default Databases;
