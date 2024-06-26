import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const HandleOutliers = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const DataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_outliers";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/Handling_Outliers.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/Handling_Outliers.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/Handling_Outliers.ipynb";

  return (
    <Container fluid>
      <h1 className="my-4">Handling Outliers</h1>
      <p>
        This section explores methods to detect and manage outliers in your
        dataset, ensuring more accurate and reliable data analysis.
      </p>

      <Row>
        <Col>
          <h2 id="what-are-outliers">What Are Outliers?</h2>
          <p>
            Outliers are data points that differ significantly from other
            observations. They can arise due to variability in the measurement
            or experimental errors, and can sometimes be indicative of
            fraudulent behavior.
          </p>

          <h2 id="detecting-outliers">Detecting Outliers</h2>
          <p>
            Detecting outliers can be done using statistical methods or
            visualizations:
          </p>
          <ul>
            <li>
              <strong>Standard Deviation:</strong> Points that are more than 3
              standard deviations from the mean are often considered outliers.
            </li>
            <li>
              <strong>Interquartile Range (IQR):</strong> Data points that fall
              below Q1 - 1.5xIQR or above Q3 + 1.5xIQR are typically considered
              outliers.
            </li>
            <li>
              <strong>Box Plots:</strong> Visual method for identifying
              outliers, showing data distribution through quartiles.
            </li>
            <li>
              <strong>Scatter Plots:</strong> Help in spotting outliers in the
              context of how data points are clustered or spread out.
            </li>
          </ul>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd
import numpy as np

# Example using IQR
Q1 = df['data'].quantile(0.25)
Q3 = df['data'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['data'] < Q1 - 1.5 * IQR) | (df['data'] > Q3 + 1.5 * IQR)]
print(outliers)`}
          />

          <h2 id="managing-outliers">Managing Outliers</h2>
          <p>
            Depending on the nature and the source of the outliers, you may
            choose to remove them or adjust them:
          </p>
          <CodeBlock
            language={"python"}
            code={`# Removing outliers
filtered_df = df[~((df['data'] < (Q1 - 1.5 * IQR)) | (df['data'] > (Q3 + 1.5 * IQR)))]

# Adjusting outliers by capping
df.loc[df['data'] < (Q1 - 1.5 * IQR), 'data'] = Q1 - 1.5 * IQR
df.loc[df['data'] > (Q3 + 1.5 * IQR), 'data'] = Q3 + 1.5 * IQR`}
          />

          <h2 id="considerations">Considerations</h2>
          <p>
            It's important to understand the impact of modifying outliers within
            your dataset. Removing or adjusting outliers without proper analysis
            can lead to biased results, especially in small datasets or datasets
            with natural variability.
          </p>

          <h2 id="validation">Validation After Handling Outliers</h2>
          <p>
            Post-processing validation is crucial to ensure the quality and
            consistency of the dataset:
          </p>
          <CodeBlock
            language={"python"}
            code={`# Visualization to confirm outlier handling
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=filtered_df['data'])
plt.title('Box Plot after Handling Outliers')
plt.show()`}
          />
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

export default HandleOutliers;
