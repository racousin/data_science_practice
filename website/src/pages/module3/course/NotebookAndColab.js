import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const NotebookAndColab = () => {
  return (
    <Container fluid>
      <h1 className="my-4">
        Interactive Data Analysis with Notebooks and Colab
      </h1>
      <p>
        Discover how to leverage Jupyter Notebooks and Google Colab for dynamic
        data analysis, enabling interactive coding, visualization, and sharing.
      </p>

      <Row>
        <Col md={12}>
          <h2 id="jupyter-notebooks">Jupyter Notebooks</h2>
          <p>
            Jupyter Notebooks provide an interactive interface to combine
            executable code, rich text, and visualizations into a single
            document. They are extensively used in data science for exploratory
            analysis, data cleaning, statistical modeling, and visualization.
          </p>

          <h3>Features of Jupyter Notebooks</h3>
          <ul>
            <li>
              Support for over 40 programming languages, including Python, R,
              and Scala.
            </li>
            <li>
              Integration with big data tools like Apache Spark from Python, R,
              and Scala kernels.
            </li>
            <li>
              Ability to share notebooks with others via email, Dropbox, GitHub
              and the Jupyter Notebook Viewer.
            </li>
          </ul>

          <h3>Installing Jupyter Notebooks</h3>
          <p>
            To get started with Jupyter Notebooks on your local machine, install
            it using pip:
          </p>
          <CodeBlock code={`pip install notebook`} />

          <h3>Launching Jupyter Notebooks</h3>
          <p>
            After installation, you can start the Jupyter Notebook by running:
          </p>
          <CodeBlock code={`jupyter notebook`} />
          <p>This command will open Jupyter in your default web browser.</p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col md={12}>
          <h2 id="google-colab">Google Colab</h2>
          <p>
            Google Colab is a cloud-based version of the Jupyter notebook
            designed for machine learning education and research. It provides a
            platform to write and execute arbitrary Python code through the
            browser, and is especially well-suited for machine learning, data
            analysis, and education.
          </p>

          <h3>Advantages of Google Colab</h3>
          <ul>
            <li>Free access to computing resources including GPUs and TPUs.</li>
            <li>No setup required to use Python libraries.</li>
            <li>Easy sharing and collaboration features.</li>
          </ul>

          <h3>Getting Started with Google Colab</h3>
          <p>
            To begin using Google Colab, navigate to the Colab website and start
            a new project:
          </p>
          <a
            href="https://colab.research.google.com/"
            target="_blank"
            rel="noopener noreferrer"
          >
            https://colab.research.google.com/
          </a>
        </Col>
      </Row>
    </Container>
  );
};

export default NotebookAndColab;
