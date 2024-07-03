import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import DataInteractionPanel from "components/DataInteractionPanel";
import CodeBlock from "components/CodeBlock";

const ExploratoryDataAnalysisAndModelBaseline = () => {
  const trainDataUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_course_train.csv";
  const testDataUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_course_test.csv";
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_requirements.txt";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_course.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_course.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module3/course/module3_course.ipynb";

  return (
    <Container fluid>
      <h1 className="my-4">Exploratory Data Analysis and Model Baseline</h1>

      <section>
        <h2 id="importance-objectives">Importance and Objectives of EDA</h2>
        <p>
          Exploratory Data Analysis (EDA) is a crucial step in the data science
          process. It allows us to:
        </p>
        <ul>
          <li>Understand the structure and characteristics of the data</li>
          <li>
            Identify patterns, trends, and relationships between variables
          </li>
          <li>Detect anomalies, outliers, and missing data</li>
          <li>Formulate hypotheses and guide further analysis</li>
        </ul>
      </section>

      <section>
        <h2 id="eda-techniques">EDA Techniques</h2>
        <h3>Univariate Analysis</h3>
        <p>Examining individual variables independently:</p>
        <CodeBlock
          language="python"
          code={`
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Histogram for a numerical variable
plt.figure(figsize=(10, 6))
df['numerical_column'].hist()
plt.title('Distribution of Numerical Column')
plt.show()

# Bar plot for a categorical variable
df['categorical_column'].value_counts().plot(kind='bar')
plt.title('Distribution of Categorical Column')
plt.show()
          `}
        />

        <h3>Bivariate Analysis</h3>
        <p>Examining relationships between pairs of variables:</p>
        <CodeBlock
          language="python"
          code={`
import seaborn as sns

# Scatter plot for two numerical variables
plt.figure(figsize=(10, 6))
sns.scatterplot(x='numerical_column1', y='numerical_column2', data=df)
plt.title('Relationship between Two Numerical Variables')
plt.show()

# Box plot for a numerical variable across categories
plt.figure(figsize=(10, 6))
sns.boxplot(x='categorical_column', y='numerical_column', data=df)
plt.title('Distribution of Numerical Column Across Categories')
plt.show()
          `}
        />

        <h3>Multivariate Analysis</h3>
        <p>Examining relationships between multiple variables:</p>
        <CodeBlock
          language="python"
          code={`
# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pair plot
sns.pairplot(df)
plt.show()
          `}
        />
      </section>

      <section>
        <h2 id="visualization-tools">Visualization Tools for EDA</h2>
        <p>Python offers several powerful libraries for data visualization:</p>
        <ul>
          <li>
            <strong>Matplotlib:</strong> The foundation for most Python plotting
            libraries
          </li>
          <li>
            <strong>Seaborn:</strong> Statistical data visualization based on
            matplotlib
          </li>
          <li>
            <strong>Plotly:</strong> Interactive, publication-quality graphs
          </li>
        </ul>
        <CodeBlock
          language="python"
          code={`
import plotly.express as px

# Interactive scatter plot
fig = px.scatter(df, x='numerical_column1', y='numerical_column2', color='categorical_column')
fig.show()
          `}
        />
      </section>

      <section>
        <h2 id="statistical-measures">Statistical Measures in EDA</h2>
        <p>Key statistical measures help summarize and understand the data:</p>
        <CodeBlock
          language="python"
          code={`
# Summary statistics
print(df.describe())

# Skewness and Kurtosis
print(df.skew())
print(df.kurtosis())

# Correlation matrix
print(df.corr())
          `}
        />
      </section>

      <section>
        <h2 id="handling-data-issues">Handling Missing Data and Outliers</h2>
        <CodeBlock
          language="python"
          code={`
# Check for missing values
print(df.isnull().sum())

# Handle missing values (example: fill with mean)
df_filled = df.fillna(df.mean())

# Detect outliers using IQR method
Q1 = df['numerical_column'].quantile(0.25)
Q3 = df['numerical_column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['numerical_column'] < lower_bound) | (df['numerical_column'] > upper_bound)]
          `}
        />
      </section>

      <section>
        <h2 id="eda-data-types">EDA for Different Data Types</h2>
        <p>EDA techniques vary depending on the type of data:</p>
        <ul>
          <li>
            <strong>Numerical data:</strong> Histograms, box plots, scatter
            plots
          </li>
          <li>
            <strong>Categorical data:</strong> Bar plots, pie charts, count
            plots
          </li>
          <li>
            <strong>Time series data:</strong> Line plots, seasonal
            decomposition
          </li>
        </ul>
      </section>

      <section>
        <h2 id="feature-engineering">From EDA to Feature Engineering</h2>
        <p>EDA often leads to insights that guide feature engineering:</p>
        <CodeBlock
          language="python"
          code={`
# Example: Creating a new feature based on EDA insights
df['new_feature'] = df['feature1'] / df['feature2']

# Example: Binning a continuous variable
df['binned_feature'] = pd.cut(df['continuous_feature'], bins=5)
          `}
        />
      </section>

      <section>
        <h2 id="jupyter-notebooks">Jupyter Notebooks</h2>
        <p>Jupyter Notebooks provide an interactive environment for EDA:</p>
        <ul>
          <li>Combine code execution with markdown documentation</li>
          <li>Visualize results inline</li>
          <li>Easy to share and collaborate</li>
        </ul>
      </section>

      <section>
        <h2 id="google-colab">Google Colab</h2>
        <p>Google Colab offers a cloud-based Jupyter Notebook environment:</p>
        <ul>
          <li>Free access to GPUs and TPUs</li>
          <li>Easy sharing and collaboration</li>
          <li>Integration with Google Drive</li>
        </ul>
      </section>

      <section>
        <h2 id="case-study">EDA and Model Baseline Case Study</h2>
        <p>
          Let's walk through a basic EDA and model baseline process using the
          provided dataset:
        </p>
        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          requirementsUrl={requirementsUrl}
        />
        <p>The notebook above demonstrates:</p>
        <ul>
          <li>Loading and initial exploration of the dataset</li>
          <li>Visualization of key variables and relationships</li>
          <li>Handling of missing data and outliers</li>
          <li>Feature engineering based on EDA insights</li>
          <li>Building and evaluating a simple baseline model</li>
        </ul>
      </section>
    </Container>
  );
};

export default ExploratoryDataAnalysisAndModelBaseline;
