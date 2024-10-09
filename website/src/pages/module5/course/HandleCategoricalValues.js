import React from 'react';
import { Container, Title, Text, Stack, List, Group, Paper } from '@mantine/core';
import { IconChartBar, IconHash, IconTags, IconBinaryTree } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import DataInteractionPanel from 'components/DataInteractionPanel';

const HandleCategoricalValues = () => {
  const metadata = {
    description: "This dataset includes characteristics of different mushrooms, aiming to classify them as either poisonous or edible based on various physical attributes.",
    source: "Mycology Research Data",
    target: "class",
    listData: [
      {
        name: "class",
        description: "Indicates whether the mushroom is poisonous (p) or edible (e).",
        dataType: "Categorical",
        example: "p (poisonous), e (edible)",
      },
      {
        name: "cap-diameter",
        description: "Numerical measurement likely representing the diameter of the mushroom's cap.",
        dataType: "Continuous",
        example: "15 cm",
      },
      {
        name: "cap-shape",
        description: "Descriptive categories for the shape of the mushroom cap (e.g., x for convex, f for flat).",
        dataType: "Categorical",
        example: "x (convex)",
      },
      {
        name: "stem-width",
        description: "Numerical measurement likely representing the width of the mushroom's stem.",
        dataType: "Continuous",
        example: "2 cm",
      },
      {
        name: "has-ring",
        description: "Boolean indicating the presence of a ring (t for true, f for false).",
        dataType: "Boolean",
        example: "t (true)",
      },
    ],
  };

  return (
    <Container fluid>
      <Title order={1} id="handle-categorical-values" mt="xl" mb="md">Handling Categorical Values</Title>
      
      <Stack spacing="xl">
        <Section
          icon={<IconTags size={24} />}
          title="Types of Categorical Data"
          id="types-of-categorical-data"
        >
          <Text>
            Categorical data can be classified into two main types:
          </Text>
          <List>
            <List.Item><strong>Nominal:</strong> Categories without inherent order (e.g., colors, types of cuisine)</List.Item>
            <List.Item><strong>Ordinal:</strong> Categories with a logical order (e.g., rankings, education level)</List.Item>
          </List>
        </Section>

        <Section
          icon={<IconChartBar size={24} />}
          title="Identify and Visualize Categorical Values"
          id="identify-and-visualize-categorical-values"
        >
          <Text>
            Identifying and visualizing categorical data is crucial for understanding distribution and influences within the dataset. Use techniques like count plots and bar plots to gain insights.
          </Text>
          <CodeBlock
            language="python"
            code={`
import seaborn as sns
import matplotlib.pyplot as plt

# Identify categorical columns
categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() < 10]

# Visualize with count plot
plt.figure(figsize=(10, 6))
sns.countplot(x='category_column', data=df, palette='Blues')
plt.title('Count Plot of Categorical Data')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualize relationship with target variable
plt.figure(figsize=(12, 8))
sns.barplot(x='category_column', y='target_column', data=df, ci=None, palette='viridis')
plt.title('Bar Plot: Categories vs Target')
plt.xlabel('Categories')
plt.ylabel('Average of Target Variable')
plt.xticks(rotation=45)
plt.show()
            `}
          />
        </Section>

        <Group grow align="flex-start" spacing="xl">
          <Section
            icon={<IconHash size={24} />}
            title="One-Hot Encoding"
            id="one-hot-encoding"
          >
            <Text>
              One-hot encoding creates a new binary column for each category. It's useful for nominal data and models expecting numerical input.
            </Text>
            <CodeBlock
              language="python"
              code={`
import pandas as pd

df = pd.get_dummies(df, columns=['category_column'])
              `}
            />
          </Section>

          <Section
            icon={<IconBinaryTree size={24} />}
            title="Label Encoding"
            id="label-encoding"
          >
            <Text>
              Label encoding converts each category to a number. It's particularly useful for ordinal data where the order matters.
            </Text>
            <CodeBlock
              language="python"
              code={`
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category_column'] = le.fit_transform(df['category_column'])
              `}
            />
          </Section>
        </Group>

        <Section
          title="Advanced Techniques"
          id="advanced-techniques"
        >
          <Stack spacing="md">
            <Paper withBorder p="md">
              <Title order={4}>Handling Unseen Categories</Title>
              <Text>
                For categories in the test set not present during training, consider assigning an 'unknown' label or using the most frequent category.
              </Text>
            </Paper>
            
            <Paper withBorder p="md">
              <Title order={4}>Feature Engineering</Title>
              <Text>
                Enhance your model by combining categories, creating interaction terms, or extracting information from mixed data types.
              </Text>
            </Paper>
            
            <Paper withBorder p="md">
              <Title order={4}>Using Categorical Data in Models</Title>
              <Text>
                Some algorithms (e.g., CatBoost, decision trees) can handle categorical data directly without explicit encoding.
              </Text>
            </Paper>
          </Stack>
        </Section>
      </Stack>

      <DataInteractionPanel
        DataUrl="/modules/module5/course/module5_course_handling_categorical.csv"
        notebookUrl="/modules/module5/course/handling_categorical.ipynb"
        notebookHtmlUrl="/modules/module5/course/handling_categorical.html"
        notebookColabUrl="/website/public/modules/module5/course/handling_categorical.ipynb"
        requirementsUrl="/modules/module5/course/module5_requirements.txt"
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

export default HandleCategoricalValues;