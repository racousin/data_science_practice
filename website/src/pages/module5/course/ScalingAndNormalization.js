import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const ScalingAndNormalization = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const DataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_scaling_and_normalization";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/Scaling_And_Normalization.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/Scaling_And_Normalization.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/Scaling_And_Normalization.ipynb";

  return (
    <Container fluid>
      <h1 className="my-4">Scaling and Normalization</h1>
      <p>
        Scaling and normalization are critical preprocessing steps in many data
        science workflows. They help to standardize the range of features, which
        is crucial for models that are sensitive to the scale of input data,
        such as distance-based algorithms.
      </p>

      <Row>
        <Col>
          <h2 id="why-scale-and-normalize">Why Scale and Normalize?</h2>
          <p>
            Different features of data often vary widely in magnitudes, units,
            and range. Without proper scaling or normalization:
          </p>
          <ul>
            <li>
              Algorithms that calculate distances between data points can be
              disproportionately influenced by one feature.
            </li>
            <li>
              Gradient descent-based algorithms may take longer to converge.
            </li>
            <li>
              Features with larger ranges could dominate the model's learning,
              leading to suboptimal performance.
            </li>
          </ul>

          <h2 id="scaling-methods">Scaling Methods</h2>
          <p>Common scaling methods include:</p>
          <ul>
            <li>
              <strong>Min-Max Scaling:</strong> Scales features to a given
              range, usually 0 to 1.
            </li>
            <li>
              <strong>Standard Scaling:</strong> Scales features to have zero
              mean and a variance of one.
            </li>
            <li>
              <strong>MaxAbs Scaling:</strong> Scales each feature by its
              maximum absolute value to be between -1 and 1 without
              shifting/centering the data.
            </li>
          </ul>
          <CodeBlock
            language={"python"}
            code={`from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

# Min-Max Scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Standard Scaling
scaler = StandardScaler()
data_standard_scaled = scaler.fit_transform(data)

# MaxAbs Scaling
scaler = MaxAbsScaler()
data_maxabs_scaled = scaler.fit_transform(data)`}
          />

          <h2 id="normalization-methods">Normalization Methods</h2>
          <p>
            Normalization adjusts the data to have a unit norm, which is often
            required in text processing and when using certain algorithms:
          </p>
          <ul>
            <li>
              <strong>L1 Normalization:</strong> Also known as least absolute
              deviations, ensures the sum of absolute values is 1 per row.
            </li>
            <li>
              <strong>L2 Normalization:</strong> Also known as least squares,
              ensures the sum of squares is 1 per row.
            </li>
          </ul>
          <CodeBlock
            language={"python"}
            code={`from sklearn.preprocessing import Normalizer

# L1 Normalization
normalizer = Normalizer(norm='l1')
data_l1_normalized = normalizer.fit_transform(data)

# L2 Normalization
normalizer = Normalizer(norm='l2')
data_l2_normalized = normalizer.fit_transform(data)`}
          />

          <h2 id="choosing-the-right-technique">
            Choosing the Right Technique
          </h2>
          <p>
            The choice between scaling and normalization techniques depends on
            the model you are using and the nature of your data. It’s essential
            to experiment with different preprocessing methods to determine
            which yields the best performance for your specific dataset.
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

export default ScalingAndNormalization;
