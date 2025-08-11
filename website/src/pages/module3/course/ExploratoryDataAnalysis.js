import React from "react";
import { Row, Col } from 'react-bootstrap';
import { Container, Grid } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
const ExploratoryDataAnalysis = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exploratory Data Analysis (EDA)</h1>
      <p>
        Exploratory Data Analysis (EDA) is the approach to analyzing datasets to summarize their main characteristics, often using visual methods. It's an important step in the data science process that precedes formal modeling or hypothesis testing.
      </p>
      <p><strong>Key Objectives:</strong></p>
      <ul>
        <li>Understand the structure and characteristics of the data</li>
        <li>Identify patterns, trends, and relationships between variables</li>
        <li>Detect anomalies, outliers, and missing data</li>
        <li>Formulate hypotheses and guide further analysis</li>
      </ul>
      <h2 id="main-components">Main Components of EDA</h2>
      <ul>
        <li>Start with simple visualizations and summary statistics</li>
        <li>Be systematic in your approach, examining all variables</li>
        <li>Look for patterns and anomalies in the data</li>
        <li>Document your findings and hypotheses</li>
        <li>Iterate between EDA and other stages of the data science process</li>
      </ul>
      <h2 id="jupyter-notebooks">Jupyter Notebooks</h2>
      <p>
        Jupyter Notebooks provide an interactive interface to combine executable code, rich text, and visualizations into a single document. They are extensively used in data science for exploratory analysis, data cleaning, statistical modeling, and visualization.
      </p>
      <h3>Installing Jupyter Notebooks</h3>
      <CodeBlock language="bash" code="pip install notebook" />
      <h3>Launching Jupyter Notebooks</h3>
      <CodeBlock language="bash" code="jupyter notebook" />
      <p>This command will open Jupyter in your default web browser.</p>
      <h2 id="google-colab">Google Colab</h2>
      <p>
        Google Colab is a cloud-based version of Jupyter notebooks designed for machine learning education and research. It provides a platform to write and execute arbitrary Python code through the browser.
      </p>
      <h3>Advantages of Google Colab</h3>
      <ul>
        <li>Free access to computing resources including GPUs and TPUs</li>
        <li>No setup required to use Python libraries</li>
        <li>Easy sharing and collaboration features</li>
      </ul>
      <h3>Getting Started with Google Colab</h3>
      <p>
        To begin using Google Colab, navigate to the Colab website and start a new project:
      </p>
      <a href="https://colab.research.google.com/" target="_blank" rel="noopener noreferrer">
        https://colab.research.google.com/
      </a>
    </Container>
  );
};
export default ExploratoryDataAnalysis;