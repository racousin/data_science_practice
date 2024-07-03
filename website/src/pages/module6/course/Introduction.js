import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Tabular Models</h1>

      <section>
        <h2 id="definition">Definition and Characteristics of Tabular Data</h2>
        <p>
          Tabular data is a type of structured data that is organized into rows
          and columns. Each row typically represents a single observation or
          instance, while each column represents a feature or attribute of those
          observations.
        </p>
        <p>Key characteristics of tabular data include:</p>
        <ul>
          <li>Organized in a table-like structure</li>
          <li>
            Each column has a specific data type (e.g., numeric, categorical,
            datetime)
          </li>
          <li>Rows are independent of each other</li>
          <li>Can be easily stored in databases or spreadsheets</li>
        </ul>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Example of tabular data
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Chicago']
}

df = pd.DataFrame(data)
print(df)
          `}
        />
      </section>

      <section>
        <h2 id="supervised-learning">
          Overview of Supervised Learning for Tabular Data
        </h2>
        <p>
          Supervised learning is a machine learning paradigm where models are
          trained on labeled data. For tabular data, this typically involves:
        </p>
        <ul>
          <li>A set of input features (X) organized in a table format</li>
          <li>A target variable (y) that we want to predict</li>
          <li>
            A model that learns to map the input features to the target variable
          </li>
        </ul>
        <p>
          There are two main types of supervised learning problems for tabular
          data:
        </p>
        <ol>
          <li>
            <strong>Regression:</strong> Predicting a continuous numerical value
          </li>
          <li>
            <strong>Classification:</strong> Predicting a categorical label
          </li>
        </ol>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate sample data
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print(f"Model coefficient: {model.coef_[0][0]:.2f}")
print(f"Model intercept: {model.intercept_[0]:.2f}")
          `}
        />
      </section>

      <section>
        <h2 id="importance">Importance of Tabular Models in Data Science</h2>
        <p>
          Tabular models play a crucial role in data science for several
          reasons:
        </p>
        <ol>
          <li>
            <strong>Ubiquity of tabular data:</strong> Many real-world datasets
            come in tabular format, including financial data, customer
            information, and scientific measurements.
          </li>
          <li>
            <strong>Interpretability:</strong> Many tabular models (e.g., linear
            regression, decision trees) provide interpretable results, which is
            crucial in fields like healthcare and finance.
          </li>
          <li>
            <strong>Efficiency:</strong> Tabular models are often
            computationally efficient, making them suitable for large datasets
            and real-time predictions.
          </li>
          <li>
            <strong>Versatility:</strong> Tabular models can handle a wide range
            of problem types, from simple linear relationships to complex
            non-linear patterns.
          </li>
          <li>
            <strong>Feature importance:</strong> Many tabular models provide
            insights into feature importance, helping in feature selection and
            understanding key drivers of predictions.
          </li>
        </ol>
        <p>Examples of real-world applications of tabular models include:</p>
        <ul>
          <li>Credit scoring in financial institutions</li>
          <li>Customer churn prediction in telecommunications</li>
          <li>Disease risk assessment in healthcare</li>
          <li>Demand forecasting in retail</li>
          <li>Fraud detection in insurance claims</li>
        </ul>
      </section>

      <section>
        <h2>Challenges in Working with Tabular Data</h2>
        <p>
          While tabular data is common and versatile, it comes with its own set
          of challenges:
        </p>
        <ul>
          <li>
            <strong>Missing values:</strong> Real-world tabular data often
            contains missing values that need to be handled appropriately.
          </li>
          <li>
            <strong>Categorical variables:</strong> Many tabular datasets
            include categorical variables that require encoding before model
            training.
          </li>
          <li>
            <strong>Feature engineering:</strong> Creating meaningful features
            from raw tabular data can significantly impact model performance.
          </li>
          <li>
            <strong>Imbalanced data:</strong> In classification problems, class
            imbalance is common and needs to be addressed.
          </li>
          <li>
            <strong>Scaling:</strong> Features in tabular data often have
            different scales and may require normalization or standardization.
          </li>
        </ul>
        <CodeBlock
          language="python"
          code={`
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample data with missing values and categorical variables
data = {
    'age': [25, 30, np.nan, 40],
    'income': [50000, 60000, 75000, np.nan],
    'city': ['New York', 'San Francisco', 'Chicago', 'New York']
}

df = pd.DataFrame(data)

# Define preprocessing steps
numeric_features = ['age', 'income']
categorical_features = ['city']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the data
X_processed = preprocessor.fit_transform(df)

print("Processed data shape:", X_processed.shape)
          `}
        />
      </section>

      <section>
        <h2>Conclusion</h2>
        <p>
          Tabular models form the backbone of many machine learning applications
          in data science. Understanding how to work with tabular data,
          including preprocessing, feature engineering, and model selection, is
          crucial for any data scientist. In the following sections of this
          module, we'll dive deeper into various tabular models, their
          implementations, and advanced techniques for model optimization and
          selection.
        </p>
      </section>
    </Container>
  );
};

export default Introduction;
