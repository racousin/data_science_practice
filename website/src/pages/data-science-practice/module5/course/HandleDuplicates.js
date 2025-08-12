import React from "react";
import { Container, Stack, Title, Text, List } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const HandleDuplicates = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const trainDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_duplicate_train.csv";
  const testDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_duplicate_test.csv";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/handling_duplicate.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/handling_duplicate.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/handling_duplicate.ipynb";
  const metadata = {
    description:
      "This dataset contains detailed meteorological measurements from various weather stations, capturing daily climatic conditions aimed at aiding weather forecasting and climatic research.",
    source: "National Weather Service",
    target: "Precipit", // Assuming Precipitation is a key measure to predict or analyze; adjust if different.
    listData: [
      {
        name: "id",
        description: "Unique identifier for each record.",
        dataType: "Integer",
        example: "1",
      },
      {
        name: "station_id",
        description: "Identifier for the weather station.",
        dataType: "Integer",
        example: "1004",
      },
      {
        name: "Date",
        description: "Date of observation.",
        dataType: "Date",
        example: "2018-02-26",
      },
      {
        name: "Temp_max",
        description:
          "Maximum temperature recorded on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "67.97°F",
      },
      {
        name: "Temp_avg",
        description: "Average temperature on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "60.39°F",
      },
      {
        name: "Temp_min",
        description:
          "Minimum temperature recorded on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "55.23°F",
      },
      {
        name: "Dew_max",
        description: "Maximum dew point on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "55.10°F",
      },
      {
        name: "Dew_avg",
        description: "Average dew point on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "59.39°F",
      },
      {
        name: "Dew_min",
        description: "Minimum dew point on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "54.76°F",
      },
      {
        name: "Hum_max",
        description: "Maximum humidity recorded on the day, in percentage.",
        dataType: "Continuous",
        example: "96.85%",
      },
      {
        name: "Hum_avg",
        description: "Average humidity on the day, in percentage.",
        dataType: "Continuous",
        example: "80.60%",
      },
      {
        name: "Hum_min",
        description: "Minimum humidity on the day, in percentage.",
        dataType: "Continuous",
        example: "60.21%",
      },
      {
        name: "Wind_max",
        description:
          "Maximum wind speed recorded on the day, in miles per hour.",
        dataType: "Continuous",
        example: "12.94 mph",
      },
      {
        name: "Wind_avg",
        description: "Average wind speed on the day, in miles per hour.",
        dataType: "Continuous",
        example: "7.71 mph",
      },
      {
        name: "Wind_min",
        description: "Minimum wind speed on the day, in miles per hour.",
        dataType: "Continuous",
        example: "5.12 mph",
      },
      {
        name: "Press_max",
        description:
          "Maximum atmospheric pressure on the day, in inches of mercury.",
        dataType: "Continuous",
        example: "30.67 inHg",
      },
      {
        name: "Press_avg",
        description:
          "Average atmospheric pressure on the day, in inches of mercury.",
        dataType: "Continuous",
        example: "28.96 inHg",
      },
      {
        name: "Press_min",
        description:
          "Minimum atmospheric pressure on the day, in inches of mercury.",
        dataType: "Continuous",
        example: "29.63 inHg",
      },
      {
        name: "Precipit",
        description: "Total precipitation on the day, in inches.",
        dataType: "Continuous",
        example: "1.01 inches",
      },
    ],
  };

  return (
    <Container fluid>
      <Title order={1} my="md">Handling Duplicates</Title>

      <Stack>
        <Title order={2} id="types-of-duplicates">Types of Duplicates</Title>
        <Text>
          Understanding the nature of duplicates is crucial for effective data cleaning:
        </Text>
        <List>
          <List.Item>
            <Text><span style={{ fontWeight: 700 }}>Exact Duplicates:</span> Records that are identical
            across all features. Often arise from data entry errors or data merging processes.</Text>
          </List.Item>
          <List.Item>
            <Text><span style={{ fontWeight: 700 }}>Partial Duplicates:</span> Records that are identical in
            key fields but differ in others. They may occur due to inconsistent data collection or merging of similar datasets.</Text>
          </List.Item>
          <List.Item>
            <Text><span style={{ fontWeight: 700 }}>Approximate Duplicates:</span> Records that are not
            identical but very similar, often due to typos or different data entry standards.</Text>
          </List.Item>
        </List>

        <Title order={2} id="identifying-duplicates">Identifying Duplicates</Title>
        <Text>
          The first step in handling duplicates is identifying them through various methods depending on their nature.
        </Text>
        <CodeBlock
          language="python"
          code={`import pandas as pd
import numpy as np

# Create a sample dataset
df = pd.DataFrame({
    'ID': [1, 2, 3, 3, 4, 5, 5],
    'Name': ['John', 'Jane', 'Bob', 'Bob', 'Alice', 'Charlie', 'Charlie'],
    'Age': [25, 30, 35, 35, 40, 45, 46],
    'City': ['New York', 'London', 'Paris', 'Paris', 'Tokyo', 'Sydney', 'Sydney']
})

# Identify exact duplicates
exact_duplicates = df[df.duplicated()]
print("Exact duplicates:")
print(exact_duplicates)

# Identify partial duplicates based on 'Name' and 'City'
partial_duplicates = df[df.duplicated(subset=['Name', 'City'], keep=False)]
print("Partial duplicates (based on Name and City):")
print(partial_duplicates)

# Count duplicates
print(f"\nNumber of exact duplicates: {exact_duplicates.shape[0]}")
print(f"Number of partial duplicates: {partial_duplicates.shape[0]}")

# Output:
# Exact duplicates:
#    ID Name  Age   City
# 3   3  Bob   35  Paris

# Partial duplicates (based on Name and City):
#    ID    Name  Age    City
# 2   3     Bob   35   Paris
# 3   3     Bob   35   Paris
# 5   5  Charlie   45  Sydney
# 6   5  Charlie   46  Sydney

# Number of exact duplicates: 1
# Number of partial duplicates: 4`}
        />

        <Title order={2} id="visualize-duplicates">Visualize Duplicates</Title>
        <Text>
          Visualizing duplicates can provide insightful perspectives on the distribution and impact of duplicate data within your dataset.
        </Text>
        <CodeBlock
          language="python"
          code={`import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Using the same sample dataset as before
df['is_duplicate'] = df.duplicated(subset=['Name', 'City'], keep=False)

# Plotting duplicates
plt.figure(figsize=(10, 6))
sns.countplot(x='is_duplicate', data=df)
plt.title('Visualization of Duplicate Records')
plt.xlabel('Is Duplicate')
plt.ylabel('Count')
plt.show()

# Visualize duplicates by category
plt.figure(figsize=(12, 6))
sns.countplot(x='City', hue='is_duplicate', data=df)
plt.title('Duplicate Records by City')
plt.xlabel('City')
plt.ylabel('Count')
plt.legend(title='Is Duplicate', labels=['No', 'Yes'])
plt.show()

# Clean up
df.drop(columns=['is_duplicate'], inplace=True)
`}
        />

        <Title order={2} id="removing-duplicates">Removing Duplicates</Title>
        <Text>
          Removing duplicates should be tailored based on the type identified and the specific needs of your dataset:
        </Text>
        <CodeBlock
          language="python"
          code={`# Remove exact duplicates
df_no_exact_dupes = df.drop_duplicates()
print("After removing exact duplicates:")
print(df_no_exact_dupes)

# Remove partial duplicates based on 'Name' and 'City', keeping the last occurrence
df_no_partial_dupes = df.drop_duplicates(subset=['Name', 'City'], keep='last')
print("After removing partial duplicates (keeping last):")
print(df_no_partial_dupes)

# Output:
# After removing exact duplicates:
#    ID    Name  Age    City
# 0   1    John   25   New York
# 1   2    Jane   30   London
# 2   3     Bob   35   Paris
# 4   4   Alice   40   Tokyo
# 5   5  Charlie  45   Sydney
# 6   5  Charlie  46   Sydney

# After removing partial duplicates (keeping last):
#    ID    Name  Age    City
# 0   1    John   25   New York
# 1   2    Jane   30   London
# 3   3     Bob   35   Paris
# 4   4   Alice   40   Tokyo
# 6   5  Charlie  46   Sydney`}
        />

        <Title order={2} id="advanced-techniques">Advanced Techniques</Title>
        <Text>
          For more complex scenarios, such as approximate duplicates, advanced techniques like fuzzy matching might be required:
        </Text>
        <CodeBlock
          language="python"
          code={`from fuzzywuzzy import process, fuzz

# Sample data with typos
names = ['John Smith', 'Jane Doe', 'John Smyth', 'Jane Do', 'Bob Johnson']

# Function to find fuzzy duplicates
def find_fuzzy_duplicates(names, threshold=80):
    duplicates = []
    for i, name in enumerate(names):
        matches = process.extract(name, names, limit=len(names), scorer=fuzz.token_sort_ratio)
        for match in matches:
            if match[1] >= threshold and match[0] != name:
                duplicates.append((name, match[0], match[1]))
    return duplicates

# Find fuzzy duplicates
fuzzy_dupes = find_fuzzy_duplicates(names)

print("Potential fuzzy duplicates:")
for original, duplicate, score in fuzzy_dupes:
    print(f"{original} <-> {duplicate} (Similarity: {score}%)")

# Output:
# Potential fuzzy duplicates:
# John Smith <-> John Smyth (Similarity: 91%)
# John Smyth <-> John Smith (Similarity: 91%)
# Jane Doe <-> Jane Do (Similarity: 89%)
# Jane Do <-> Jane Doe (Similarity: 89%)`}
        />

        <Title order={2} id="considerations">Considerations</Title>
        <Text>
          Consider the implications of removing duplicates in your data analysis. It's essential to understand why duplicates appear and confirm that their removal is justified:
        </Text>
      
      </Stack>
        <div id="notebook-example"></div>
        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          requirementsUrl={requirementsUrl}
          metadata={metadata}
        />
    </Container>
  );
};

export default HandleDuplicates;
