import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const FeatureExtraction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Feature Extraction and Transformation</h1>
      <p>
        In this section, you will learn about various methods for extracting and
        transforming features from raw data.
      </p>
      <Row>
        <Col>
          <h2>Polynomial Features and Interaction Terms</h2>
          <p>
            Polynomial features can be used to capture non-linear relationships
            between features. Interaction terms can be used to capture
            interactions between features.
          </p>
          <CodeBlock
            code={`# Example of adding polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)`}
          />
          <h2>Dimensionality Reduction Techniques</h2>
          <p>
            Dimensionality reduction techniques can be used to reduce the number
            of features in the data while preserving as much of the original
            information as possible.
          </p>
          <CodeBlock
            code={`# Example of PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)`}
          />
          <h2>Text Data Processing</h2>
          <p>
            Text data can be processed using techniques such as TF-IDF and word
            embeddings to convert text data into numerical data that can be used
            by machine learning algorithms.
          </p>
          <CodeBlock
            code={`# Example of TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default FeatureExtraction;
