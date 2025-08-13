import React from 'react';
import { Container, Title, Text, Stack, List, Group, Table } from '@mantine/core';
import { IconFilter, IconFold, IconChartDots3, IconChartBar } from '@tabler/icons-react';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import DataInteractionPanel from 'components/DataInteractionPanel';

const FeatureSelectionAndDimensionalityReduction = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/data-science-practice/module5/course/module5_requirements.txt";
  const dataUrl =
    process.env.PUBLIC_URL +
    "/modules/data-science-practice/module5/course/module5_course_feature_selection_and_dimensionality_reduction.csv";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/data-science-practice/module5/course/feature_selection_and_dimensionality_reduction.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL +
    "/modules/data-science-practice/module5/course/feature_selection_and_dimensionality_reduction.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/data-science-practice/module5/course/feature_selection_and_dimensionality_reduction.ipynb";

  const metadata = {
    description:
      "This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.",
    source: "Breast Cancer Wisconsin (Diagnostic) Data Set",
    target: "target (1 = malignant, 0 = benign)",
    listData: [
      {
        name: "mean radius",
        description: "Mean of distances from center to points on the perimeter",
      },
      {
        name: "mean texture",
        description: "Standard deviation of gray-scale values",
      },
      {
        name: "mean perimeter",
        description: "Mean size of the core tumor",
      },
      {
        name: "mean area",
        description: "Mean area of the core tumor",
      },
      {
        name: "mean smoothness",
        description: "Mean of local variation in radius lengths",
      },
      {
        name: "mean compactness",
        description: "Mean of perimeter^2 / area - 1.0",
      },
      {
        name: "mean concavity",
        description: "Mean of severity of concave portions of the contour",
      },
      {
        name: "mean concave points",
        description: "Mean for number of concave portions of the contour",
      },
      {
        name: "mean symmetry",
        description: "Mean symmetry of the cell nucleus",
      },
      {
        name: "mean fractal dimension",
        description: "Mean for 'coastline approximation' - 1",
      },
      {
        name: "radius error",
        description:
          "Standard error for the mean of distances from center to points on the perimeter",
      },
      {
        name: "texture error",
        description:
          "Standard error for standard deviation of gray-scale values",
      },
      {
        name: "perimeter error",
        description: "Standard error for mean size of the core tumor",
      },
      {
        name: "area error",
        description: "Standard error for mean area of the core tumor",
      },
      {
        name: "smoothness error",
        description:
          "Standard error for mean of local variation in radius lengths",
      },
      {
        name: "compactness error",
        description: "Standard error for mean of perimeter^2 / area - 1.0",
      },
      {
        name: "concavity error",
        description:
          "Standard error for mean of severity of concave portions of the contour",
      },
      {
        name: "concave points error",
        description:
          "Standard error for mean for number of concave portions of the contour",
      },
      {
        name: "symmetry error",
        description: "Standard error for mean symmetry of the cell nucleus",
      },
      {
        name: "fractal dimension error",
        description:
          "Standard error for mean for 'coastline approximation' - 1",
      },
      {
        name: "worst radius",
        description:
          "Worst or largest mean value for distance from center to points on the perimeter",
      },
      {
        name: "worst texture",
        description:
          "Worst or largest mean value for standard deviation of gray-scale values",
      },
      {
        name: "worst perimeter",
        description: "Worst or largest mean value for core tumor size",
      },
      {
        name: "worst area",
        description: "Worst or largest mean value for core tumor area",
      },
      {
        name: "worst smoothness",
        description:
          "Worst or largest mean value for local variation in radius lengths",
      },
      {
        name: "worst compactness",
        description: "Worst or largest mean value for perimeter^2 / area - 1.0",
      },
      {
        name: "worst concavity",
        description:
          "Worst or largest mean value for severity of concave portions of the contour",
      },
      {
        name: "worst concave points",
        description:
          "Worst or largest mean value for number of concave portions of the contour",
      },
      {
        name: "worst symmetry",
        description:
          "Worst or largest mean value for symmetry of the cell nucleus",
      },
      {
        name: "worst fractal dimension",
        description:
          "Worst or largest mean value for 'coastline approximation' - 1",
      },
    ],
  };
  return (
    <Container fluid>
      <Title order={1} id="feature-selection-dimensionality-reduction" mt="xl" mb="md">Feature Selection and Dimensionality Reduction</Title>
      
      <Text>
        Feature selection and dimensionality reduction are crucial techniques in machine learning that help to identify the most important features and reduce the complexity of high-dimensional datasets. These methods can improve model performance, reduce overfitting, and enhance interpretability.
      </Text>

      <Stack spacing="xl" mt="xl">
        <Section
          icon={<IconFilter size={24} />}
          title="Feature Selection Techniques"
          id="feature-selection"
        >
          <Text>
            Feature selection is the process of selecting a subset of relevant features for use in model construction. It's used to simplify models, reduce training times, and improve generalization by reducing overfitting.
          </Text>
          
          <Title order={4} mt="md">1. Filter Methods</Title>
          <Text>
            Filter methods select features based on their scores in various statistical tests for their correlation with the outcome variable.
          </Text>
          <List>
            <List.Item><span style={{ fontWeight: 700 }}>Correlation Coefficient:</span> Measures the linear relationship between two variables.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>Chi-Squared Test:</span> Measures the dependence between categorical variables.</List.Item>
          </List>
          <CodeBlock
            language="python"
            code={`
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

# Assuming X is your feature matrix and y is your target variable
selector = SelectKBest(f_classif, k=5)  # Select top 5 features
X_new = selector.fit_transform(X, y)

# Get the indices of the selected features
selected_features = selector.get_support(indices=True)
feature_names = X.columns[selected_features]
print("Selected features:", feature_names)
            `}
          />

          <Title order={4} mt="md">2. Wrapper Methods</Title>
          <Text>
            Wrapper methods use a predictive model to score feature subsets. They train a new model for each subset and are computationally intensive.
          </Text>
          <List>
            <List.Item><span style={{ fontWeight: 700 }}>Recursive Feature Elimination (RFE):</span> Recursively removes attributes and builds a model on those attributes that remain.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>Forward Feature Selection:</span> Iteratively adds the best feature to the set of selected features.</List.Item>
          </List>
          <CodeBlock
            language="python"
            code={`
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=5)
fit = rfe.fit(X, y)

print("Selected features:", X.columns[fit.support_])
            `}
          />

          <Title order={4} mt="md">3. Embedded Methods</Title>
          <Text>
            Embedded methods perform feature selection as part of the model creation process.
          </Text>
          <List>
            <List.Item><span style={{ fontWeight: 700 }}>Lasso Regression:</span> Uses L1 regularization to shrink some coefficients to zero, effectively selecting features.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>Random Forest Feature Importance:</span> Uses the tree-based structure to rank features by their importance.</List.Item>
          </List>
          <CodeBlock
            language="python"
            code={`
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

model = RandomForestClassifier()
model.fit(X, y)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
            `}
          />
        </Section>

        <Section
          icon={<IconFold size={24} />}
          title="Dimensionality Reduction Techniques"
          id="dimensionality-reduction"
        >
          <Text>
            Dimensionality reduction techniques transform the data from a high-dimensional space into a lower-dimensional space, retaining meaningful properties of the original data.
          </Text>
          
          <Title order={4} mt="md">1. Principal Component Analysis (PCA)</Title>
          <Text>
            PCA is an unsupervised linear transformation technique that finds the directions of maximum variance in high-dimensional data and projects it onto a new subspace with equal or fewer dimensions than the original.
          </Text>
          <BlockMath math="\Sigma = \frac{1}{n-1} X^T X" />
          <CodeBlock
            language="python"
            code={`
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Dataset')
plt.colorbar()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
            `}
          />

          <Title order={4} mt="md">2. t-Distributed Stochastic Neighbor Embedding (t-SNE)</Title>
          <Text>
            t-SNE is a nonlinear technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets.
          </Text>
          <BlockMath math="KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}" />
          <CodeBlock
            language="python"
            code={`
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE visualization of dataset')
plt.colorbar()
plt.show()
            `}
          />

          <Title order={4} mt="md">3. Linear Discriminant Analysis (LDA)</Title>
          <Text>
            LDA is a supervised method used for both classification and dimensionality reduction. It projects the data onto a lower-dimensional space while maximizing the separation between classes.
          </Text>
          <BlockMath math="J(w) = \frac{w^T S_B w}{w^T S_W w}" />
          <CodeBlock
            language="python"
            code={`
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.xlabel('First LDA component')
plt.ylabel('Second LDA component')
plt.title('LDA of dataset')
plt.colorbar()
plt.show()

print("Explained variance ratio:", lda.explained_variance_ratio_)
            `}
          />
        </Section>

        <Section
          icon={<IconChartDots3 size={24} />}
          title="Comparison of Techniques"
          id="comparison"
        >
          <Table>
            <thead>
              <tr>
                <th>Technique</th>
                <th>Pros</th>
                <th>Cons</th>
                <th>Best Use Cases</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Filter Methods</td>
                <td>Fast, independent of learning algorithm</td>
                <td>May not capture feature interactions</td>
                <td>Large datasets, quick initial screening</td>
              </tr>
              <tr>
                <td>Wrapper Methods</td>
                <td>Consider feature interactions, tailored to specific algorithms</td>
                <td>Computationally intensive, risk of overfitting</td>
                <td>Small to medium-sized datasets</td>
              </tr>
              <tr>
                <td>Embedded Methods</td>
                <td>Consider feature interactions, more efficient than wrapper methods</td>
                <td>Specific to certain algorithms</td>
                <td>When using algorithms that support embedded selection</td>
              </tr>
              <tr>
                <td>PCA</td>
                <td>Preserves global structure, computationally efficient</td>
                <td>Reduced interpretability, assumes linear relationships</td>
                <td>Datasets with many correlated features</td>
              </tr>
              <tr>
                <td>t-SNE</td>
                <td>Preserves local structure, effective for visualization</td>
                <td>Computationally intensive, results can vary</td>
                <td>High-dimensional data visualization</td>
              </tr>
              <tr>
                <td>LDA</td>
                <td>Maximizes class separability, can be used for classification</td>
                <td>Assumes normal distribution with equal covariance</td>
                <td>Multi-class classification problems</td>
              </tr>
            </tbody>
          </Table>
        </Section>

        <Section
          icon={<IconChartBar size={24} />}
          title="Best Practices"
          id="best-practices"
        >
          <List>
            <List.Item><span style={{ fontWeight: 700 }}>Understand Your Data:</span> Thoroughly explore your dataset before applying any technique.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>Combine Multiple Techniques:</span> Often, a combination of techniques can yield better results.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>Consider Interpretability:</span> If model interpretability is important, favor feature selection over dimensionality reduction.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>Scale Your Data:</span> Normalize or standardize your data before applying dimensionality reduction techniques.</List.Item>
          </List>
        </Section>
      </Stack>
      <div id="notebook-example"></div>
      <DataInteractionPanel
        dataUrl="/modules/data-science-practice/module5/course/module5_course_feature_selection_and_dimensionality_reduction.csv"
        notebookUrl="/modules/data-science-practice/module5/course/feature_selection_and_dimensionality_reduction.ipynb"
        notebookHtmlUrl="/modules/data-science-practice/module5/course/feature_selection_and_dimensionality_reduction.html"
        notebookColabUrl="/website/public/modules/data-science-practice/module5/course/feature_selection_and_dimensionality_reduction.ipynb"
        requirementsUrl="/modules/data-science-practice/module5/course/module5_requirements.txt"
        metadata={metadata}
      />
    </Container>
  );
};

const Section = ({ icon, title, id, children }) => (
  <Stack spacing="sm">
    <Group spacing="xs">
      {icon}
      <Title order={2} id={id}>{title}</Title>
    </Group>
    {children}
  </Stack>
);

export default FeatureSelectionAndDimensionalityReduction;