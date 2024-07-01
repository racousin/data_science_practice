import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const HandleDuplicates = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const trainDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_duplicates_train.csv";
  const testDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_duplicates_test.csv";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/Handling_Duplicates.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/Handling_Duplicates.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/Handling_Duplicates.ipynb";

  return (
    <Container fluid>
      <h1 className="my-4">Handling Duplicate Entries</h1>
      <p>
        Duplicates in a dataset can skew results and lead to inaccurate
        analyses. This section provides a comprehensive guide to identifying,
        categorizing, and removing duplicate entries effectively.
      </p>

      <Row>
        <Col>
          <h2 id="types-of-duplicates">Types of Duplicates</h2>
          <p>
            Understanding the nature of duplicates is crucial for effective data
            cleaning:
          </p>
          <ul>
            <li>
              <strong>Exact Duplicates:</strong> Records that are identical
              across all features. Often arise from data entry errors or data
              merging processes.
            </li>
            <li>
              <strong>Partial Duplicates:</strong> Records that are identical in
              key fields but differ in others. They may occur due to
              inconsistent data collection or merging of similar datasets.
            </li>
            <li>
              <strong>Approximate Duplicates:</strong> Records that are not
              identical but very similar, often due to typos or different data
              entry standards.
            </li>
          </ul>

          <h2 id="identifying-duplicates">Identifying Duplicates</h2>
          <p>
            The first step in handling duplicates is identifying them through
            various methods depending on their nature.
          </p>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd

# For exact duplicates
exact_duplicates = df[df.duplicated()]

# For partial duplicates, specify columns
partial_duplicates = df[df.duplicated(subset=['column1', 'column2'])]

print("Exact duplicates:", exact_duplicates.shape[0])
print("Partial duplicates:", partial_duplicates.shape[0])`}
          />
          <h2 id="visualize-duplicates">Visualize Duplicates</h2>
          <p>
            Visualizing duplicates can provide insightful perspectives on the
            distribution and impact of duplicate data within your dataset. This
            visualization helps in identifying patterns that might influence the
            handling strategy for duplicates, especially when deciding whether
            to remove or modify them.
          </p>
          <CodeBlock
            language={"python"}
            code={`import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
# Creating a temporary column 'is_duplicate' to mark duplicate rows
df['is_duplicate'] = df.duplicated(keep=False)

# Plotting duplicates
plt.figure(figsize=(10, 6))
sns.countplot(x='is_duplicate', data=df)
plt.title('Visualization of Duplicate Records')
plt.xlabel('Is Duplicate')
plt.ylabel('Count')
plt.show()

# Dropping the temporary column after visualization
df.drop(columns=['is_duplicate'], inplace=True)`}
          />

          <p>
            This visualization uses a simple count plot to show the presence of
            duplicate entries in the dataset. It marks each row as a duplicate
            or not and counts the occurrences, providing a clear visual
            representation of how many entries are affected. This method is
            particularly useful for quickly assessing the extent of duplication
            and determining if further cleaning steps are necessary.
          </p>

          <h2 id="removing-duplicates">Removing Duplicates</h2>
          <p>
            Removing duplicates should be tailored based on the type identified
            and the specific needs of your dataset:
          </p>
          <CodeBlock
            language={"python"}
            code={`# Removing exact duplicates
df = df.drop_duplicates()

# Keeping the last occurrence of partial duplicates
df = df.drop_duplicates(subset=['column1', 'column2'], keep='last')`}
          />

          <h2 id="advanced-techniques">Advanced Techniques</h2>
          <p>
            For more complex scenarios, such as approximate duplicates, advanced
            techniques like fuzzy matching might be required:
          </p>
          <CodeBlock
            language={"python"}
            code={`from fuzzywuzzy import process

# Example of using fuzzy matching to find close matches
choices = df['column_name'].unique()
matches = process.extract('search_term', choices, limit=10)
print(matches)`}
          />

          <h2 id="considerations">Considerations</h2>
          <p>
            Consider the implications of removing duplicates in your data
            analysis. It’s essential to understand why duplicates appear and
            confirm that their removal is justified:
          </p>
          <CodeBlock
            language={"python"}
            code={`# Considerations for time-series data
if 'date' in df.columns:
    df = df.drop_duplicates(subset=['date', 'category'], keep='first')`}
          />
        </Col>
      </Row>
      <Row>
        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          requirementsUrl={requirementsUrl}
        />
      </Row>
    </Container>
  );
};

export default HandleDuplicates;
