import React from 'react';
import { Text, Title, Paper, Grid, List, ThemeIcon, Accordion, Alert } from '@mantine/core';
import { Database, Key, GitMerge, FileSpreadsheet, Globe } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const ManipulatingSources = () => {
  return (
    <div>
      <Title order={1}>Manipulating Different Sources with pandas</Title>
      
      <Text mt="md">
        In data science, it's common to work with data from various sources. pandas is a powerful Python library that provides tools to efficiently combine and manipulate data from different sources.
      </Text>

      <Title order={2} mt="xl" id="combining-data">Combining Data from Multiple Sources</Title>
      
      <Grid mt="md">
        <Grid.Col span={4}>
          <Paper withBorder p="md">
            <ThemeIcon size={30} radius="md" variant="light" color="blue" mb="sm">
              <FileSpreadsheet size={20} />
            </ThemeIcon>
            <Title order={3}>File-based Sources</Title>
            <Text size="sm">CSV, Excel, JSON files</Text>
          </Paper>
        </Grid.Col>
        
        <Grid.Col span={4}>
          <Paper withBorder p="md">
            <ThemeIcon size={30} radius="md" variant="light" color="green" mb="sm">
              <Database size={20} />
            </ThemeIcon>
            <Title order={3}>Databases</Title>
            <Text size="sm">SQL, NoSQL databases</Text>
          </Paper>
        </Grid.Col>
        
        <Grid.Col span={4}>
          <Paper withBorder p="md">
            <ThemeIcon size={30} radius="md" variant="light" color="orange" mb="sm">
              <Globe size={20} />
            </ThemeIcon>
            <Title order={3}>APIs</Title>
            <Text size="sm">Web services, RESTful APIs</Text>
          </Paper>
        </Grid.Col>
      </Grid>

      <Title order={3} mt="xl">Steps to Combine Data</Title>
      
      <List mt="md">
        <List.Item>Load data from each source into pandas DataFrames</List.Item>
        <List.Item>Clean and preprocess each DataFrame as needed</List.Item>
        <List.Item>Identify common keys or columns for joining</List.Item>
        <List.Item>Use pandas merge, join, or concat functions to combine DataFrames</List.Item>
        <List.Item>Handle any conflicts or inconsistencies in the combined data</List.Item>
      </List>

      <Title order={2} mt="xl" id="id-management">ID Management</Title>
      
      <Text mt="md">
        Proper ID management is crucial when combining data from multiple sources. Unique identifiers (IDs) are used to accurately join and merge datasets.
      </Text>

      <Alert icon={<Key size={16} />} title="Why IDs Matter" color="blue" mt="md">
        Unique IDs ensure that data from different sources is correctly matched and combined, preventing errors and inconsistencies in your analysis.
      </Alert>

      <Accordion mt="xl">
        <Accordion.Item value="creating-ids">
          <Accordion.Control icon={<Key size={20} />}>
            Creating Unique IDs
          </Accordion.Control>
          <Accordion.Panel>
            <Text>When datasets lack a common identifier, you may need to create one:</Text>
            <List mt="sm">
              <List.Item>Use a combination of existing columns to create a unique key</List.Item>
              <List.Item>Generate a new unique identifier using pandas' built-in functions</List.Item>
            </List>
            <CodeBlock
        language="python"
        code=
{`# Creating a unique ID from multiple columns
df['unique_id'] = df['column1'] + '_' + df['column2'] + '_' + df['column3'].astype(str)

# Generate a new unique identifier
df['new_id'] = pd.Series(range(len(df)))
`}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="handling-conflicts">
          <Accordion.Control icon={<GitMerge size={20} />}>
            Handling ID Conflicts
          </Accordion.Control>
          <Accordion.Panel>
            <Text>When merging datasets, you may encounter ID conflicts:</Text>
            <List mt="sm">
              <List.Item>Use pandas merge with different join types (left, right, inner, outer)</List.Item>
              <List.Item>Handle duplicate IDs by aggregating or choosing a preferred source</List.Item>
            </List>
            <CodeBlock
        language="python"
        code=
{`# Merging with handling of duplicate IDs
merged_df = pd.merge(df1, df2, on='id', how='left', suffixes=('_primary', '_secondary'))

# Prioritize data from df1 in case of conflicts
merged_df['value'] = merged_df['value_primary'].fillna(merged_df['value_secondary'])
`}
            />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <Title order={2} mt="xl">Concrete Use Cases</Title>

      <Paper withBorder p="md" mt="md">
        <Title order={3}>Case 1: Combining Sales and Customer Data</Title>
        <Text mt="sm">
          Let's combine sales data from a CSV file with customer information from a SQL database.
        </Text>
        <CodeBlock
        language="python"
        code=
{`import pandas as pd
import sqlite3

# Load sales data from CSV
sales_df = pd.read_csv('sales_data.csv')

# Load customer data from SQL database
conn = sqlite3.connect('customer_database.db')
customer_df = pd.read_sql_query("SELECT * FROM customers", conn)

# Merge sales and customer data
merged_df = pd.merge(sales_df, customer_df, on='customer_id', how='left')

print(merged_df.head())
`}
        />
      </Paper>

      <Paper withBorder p="md" mt="xl">
        <Title order={3}>Case 2: Combining Real-time Data with Historical Records</Title>
        <Text mt="sm">
          Here's an example of how to combine real-time data from an API with historical records stored in a CSV file.
        </Text>
        <CodeBlock
        language="python"
        code=
{`import pandas as pd
import requests

# Load historical data from CSV
historical_df = pd.read_csv('historical_data.csv')

# Fetch real-time data from an API
api_url = "https://api.example.com/realtime-data"
response = requests.get(api_url)
realtime_data = response.json()

# Convert real-time data to DataFrame
realtime_df = pd.DataFrame(realtime_data)

# Combine historical and real-time data
combined_df = pd.concat([historical_df, realtime_df], ignore_index=True)

# Remove duplicates based on a timestamp column
combined_df.drop_duplicates(subset='timestamp', keep='last', inplace=True)

print(combined_df.tail())
`}
        />
      </Paper>

    </div>
  );
};

export default ManipulatingSources;