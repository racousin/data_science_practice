import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const FeatureSelectionAndReduction = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const DataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_feature_selection_and_reduction";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/Feature_Selection_And_Reduction.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/Feature_Selection_And_Reduction.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/Feature_Selection_And_Reduction.ipynb";

  return (
    <Container fluid>
      <h1 className="my-4">Feature Selection and Dimension Reduction</h1>
      <p>
        Effective feature selection and dimension reduction can significantly
        improve model performance by reducing overfitting, enhancing
        generalization, and speeding up training processes.
      </p>

      <Row>
        <Col>
          <h2 id="feature-selection">Feature Selection</h2>
          <p>
            Feature selection techniques aim to remove irrelevant or redundant
            features from data that do not contribute to the predictive power of
            the model.
          </p>
          <ul>
            <li>
              <strong>Filter Methods:</strong> Use statistical measures to score
              the relevance of features with the target variable (e.g.,
              correlation coefficient, Chi-square test).
            </li>
            <li>
              <strong>Wrapper Methods:</strong> Use a subset of features and
              train a model to evaluate their performance (e.g., recursive
              feature elimination).
            </li>
            <li>
              <strong>Embedded Methods:</strong> Perform feature selection as
              part of the model training process (e.g., Lasso, Decision Trees).
            </li>
          </ul>
          <CodeBlock
            language={"python"}
            code={`from sklearn.feature_selection import SelectKBest, chi2
# Selecting the k-best features using a chi-squared test
selector = SelectKBest(score_func=chi2, k=4)
X_new = selector.fit_transform(X, y)`}
          />

          <h2 id="dimension-reduction">Dimension Reduction</h2>
          <p>
            Dimensionality reduction techniques reduce the number of random
            variables to consider, based on obtaining a set of principal
            variables.
          </p>
          <ul>
            <li>
              <strong>Principal Component Analysis (PCA):</strong> Linear
              dimensionality reduction using Singular Value Decomposition of the
              data to project it to a lower dimensional space.
            </li>
            <li>
              <strong>
                t-Distributed Stochastic Neighbor Embedding (t-SNE):
              </strong>{" "}
              Non-linear technique particularly well suited for the
              visualization of high-dimensional datasets.
            </li>
            <li>
              <strong>Autoencoders:</strong> Neural network based approach used
              to encode the input into a lower-dimensional space and then
              reconstruct the output.
            </li>
          </ul>
          <CodeBlock
            language={"python"}
            code={`from sklearn.decomposition import PCA
# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)`}
          />

          <h2 id="applications">Applications</h2>
          <p>
            These techniques are widely applied in fields like bioinformatics
            for gene selection, image processing to reduce the number of
            features in an image, and text analysis for reducing dimensions of
            text data.
          </p>

          <h2 id="choosing-the-right-technique">
            Choosing the Right Technique
          </h2>
          <p>
            The choice of technique largely depends on the type of data and the
            specific needs of the application. Experimentation is often
            necessary to find the optimal approach.
          </p>
        </Col>
      </Row>
      <Row>
        <DataInteractionPanel
          DataUrl={DataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          requirementsUrl={requirementsUrl}
        />
      </Row>
    </Container>
  );
};

export default FeatureSelectionAndReduction;
