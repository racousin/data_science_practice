import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DataPreprocessing = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Data Preprocessing Techniques</h1>
      <p>
        In this section, you will learn about various techniques for
        preprocessing data to prepare it for feature engineering.
      </p>
      <Row>
        <Col>
          <h2>Handling Missing Values</h2>
          <p>
            Missing values can be handled using various techniques such as
            mean/median/mode imputation, k-nearest neighbors imputation, and
            predictive imputation.
          </p>
          <CodeBlock
            code={`# Example of mean imputation
import numpy as np
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X_imputed = imp.transform(X)`}
          />
          <h2>Data Scaling and Normalization</h2>
          <p>
            Data scaling and normalization can be used to ensure that all
            features have the same scale. This can improve the performance of
            machine learning algorithms that are sensitive to the scale of the
            input data.
          </p>
          <CodeBlock
            code={`# Example of min-max scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)`}
          />
          <h2>Encoding Categorical Data</h2>
          <p>
            Categorical data can be encoded using various techniques such as
            one-hot encoding, ordinal encoding, and binary encoding.
          </p>
          <CodeBlock
            code={`# Example of one-hot encoding
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)`}
          />
          <h2>Binning Continuous Variables</h2>
          <p>
            Continuous variables can be binned into discrete intervals to
            simplify the data and make it easier to model.
          </p>
          <CodeBlock
            code={`# Example of binning a continuous variable
X['age_group'] = pd.cut(X['age'], bins=[0, 20, 40, 60, 100], labels=['0-20', '20-40', '40-60', '60+'])`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default DataPreprocessing;
