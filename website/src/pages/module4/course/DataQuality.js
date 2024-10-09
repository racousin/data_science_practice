import React from 'react';
import { Text, Title, Paper, Grid, List, ThemeIcon, Accordion } from '@mantine/core';
import { CheckCircle, AlertTriangle, RefreshCw, Filter } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const DataQuality = () => {
  return (
    <div>
      <Title order={1}>Data Quality and Governance</Title>
      
      <Text mt="md" id="quality-overview">
        Ensuring high data quality is crucial for reliable analysis and decision-making in data science. Data quality refers to the condition of a set of values of qualitative or quantitative variables. High-quality data is accurate, complete, consistent, and timely.
      </Text>

      <Paper withBorder p="md" mt="xl">
        <Title order={2}>Why Data Quality Matters</Title>
        <List mt="sm">
          <List.Item>Ensures accurate insights and predictions</List.Item>
          <List.Item>Reduces errors in decision-making</List.Item>
          <List.Item>Increases efficiency in data processing</List.Item>
          <List.Item>Enhances trust in data-driven processes</List.Item>
        </List>
      </Paper>

      <Title order={2} mt="xl" id="key-aspects">Key Aspects of Data Quality</Title>
      
      <Accordion mt="md">
        <Accordion.Item value="completeness">
          <Accordion.Control icon={<CheckCircle size={20} />}>
            Completeness
          </Accordion.Control>
          <Accordion.Panel>
            Ensures all required data is available. Check for missing values and handle them appropriately.
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="consistency">
          <Accordion.Control icon={<RefreshCw size={20} />}>
            Consistency
          </Accordion.Control>
          <Accordion.Panel>
            Data should be uniform across all sources and systems. Ensure data formats and definitions are standardized.
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="accuracy">
          <Accordion.Control icon={<AlertTriangle size={20} />}>
            Accuracy
          </Accordion.Control>
          <Accordion.Panel>
            Data should correctly represent the real-world values. Verify data against trusted sources when possible.
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="validity">
          <Accordion.Control icon={<Filter size={20} />}>
            Validity
          </Accordion.Control>
          <Accordion.Panel>
            Data should conform to defined formats and fall within acceptable ranges.
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <Title order={2} mt="xl">Data Cleaning Techniques</Title>
      
      <Grid mt="md">
        <Grid.Col span={6}>
          <Paper withBorder p="md">
            <Title order={3}>Handling Missing Values</Title>
            <Text mt="sm">
              Missing values can be handled through various methods:
            </Text>
            <List mt="sm">
              <List.Item>Imputation (mean, median, mode)</List.Item>
              <List.Item>Deletion (listwise or pairwise)</List.Item>
              <List.Item>Prediction using machine learning</List.Item>
            </List>
          </Paper>
        </Grid.Col>
        
        <Grid.Col span={6}>
          <Paper withBorder p="md">
            <Title order={3}>Dealing with Outliers</Title>
            <Text mt="sm">
              Outliers can significantly impact analysis. Common approaches:
            </Text>
            <List mt="sm">
              <List.Item>Trimming</List.Item>
              <List.Item>Winsorization</List.Item>
              <List.Item>Transformation (e.g., log transformation)</List.Item>
            </List>
          </Paper>
        </Grid.Col>
      </Grid>

      <Title order={2} mt="xl">Example: Data Cleaning with Pandas</Title>
      
      <CodeBlock
        language="python"
        code=
{`import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Rename columns
df.rename(columns={'old_column_name': 'new_column_name'}, inplace=True)

# Drop corrupt columns (example: column with irrelevant or corrupt data)
df.drop(columns=['corrupt_column'], inplace=True)

# Set an index and reset it
df.set_index('new_column_name', inplace=True)
df.reset_index(drop=False, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop corrupt lines (example: drop rows with NaN in critical columns)
df.dropna(subset=['important_column'], inplace=True)

# Convert text to uppercase and remove redundant spaces
df['text_column'] = df['text_column'].str.upper().str.strip()

print(df.head())

`}
      />

      <Title order={2} mt="xl">Data Governance</Title>
      
      <Text mt="md">
        Data governance involves the overall management of data availability, usability, integrity, and security. It encompasses policies, procedures, and standards that ensure high-quality data throughout its lifecycle.
      </Text>

      <Paper withBorder p="md" mt="md">
        <Title order={3}>Key Components of Data Governance</Title>
        <List mt="sm">
          <List.Item>Data policies and standards</List.Item>
          <List.Item>Data quality management</List.Item>
          <List.Item>Data security and privacy</List.Item>
          <List.Item>Data lifecycle management</List.Item>
          <List.Item>Metadata management</List.Item>
        </List>
      </Paper>

    </div>
  );
};

export default DataQuality;