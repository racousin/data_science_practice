import React from 'react';
import { Text, Title, Paper, Stack, List } from '@mantine/core';
import { Database, FileSpreadsheet, Globe } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const ManipulatingSources = () => {
  return (
    <Stack gap="xl">
      <div>
        <Title order={1}>Manipulating Different Sources with pandas</Title>
        <Text mt="md">
          Data science projects often require combining data from multiple sources. pandas provides essential tools 
          to read, transform, and combine data from various formats into a unified dataset.
        </Text>
      </div>

      <Paper p="lg" radius="md" className="bg-slate-50">
        <Title order={2}>Reading from Different Sources</Title>
        <CodeBlock
          language="python"
          code={`# Reading from common data sources
import pandas as pd

# CSV files
df_csv = pd.read_csv('data.csv', 
                     parse_dates=['date_column'],    # Parse date columns
                     dtype={'id': str})              # Specify column types

# Excel files
df_excel = pd.read_excel('data.xlsx',
                        sheet_name='Sheet1',         # Specify sheet
                        usecols=['A:C'])            # Select columns

# JSON data
df_json = pd.read_json('data.json',
                      orient='records')             # Specify JSON structure

# SQL databases
from sqlalchemy import create_engine
engine = create_engine('database_url')
df_sql = pd.read_sql('SELECT * FROM table', engine)

# API responses
import requests
response = requests.get('api_url').json()
df_api = pd.DataFrame(response['data'])`}
        />
      </Paper>

      <Paper p="lg" radius="md" className="bg-slate-50">
        <Title order={2}>Data Type Management</Title>
        <CodeBlock
          language="python"
          code={`# Common data type operations
import pandas as pd

df = pd.DataFrame({
    'id': ['1', '2', '3'],
    'value': ['10.5', '20.0', '30.7'],
    'date': ['2024-01-01', '2024-01-02', '2024-01-03']
})

# Convert types
df['id'] = df['id'].astype(int)                    # String to integer
df['value'] = pd.to_numeric(df['value'])           # String to float
df['date'] = pd.to_datetime(df['date'])            # String to datetime

# Verify datatypes
print(df.dtypes)`}
        />
      </Paper>

      <Paper p="lg" radius="md" className="bg-slate-50">
        <Title order={2}>Combining DataFrames</Title>
        <Text mt="md">pandas offers three main ways to combine DataFrames:</Text>
        <CodeBlock
          language="python"
          code={`# Different methods to combine DataFrames
import pandas as pd

# 1. Merge: Combine based on common columns (like SQL JOIN)
df_merged = pd.merge(
    left=df1,
    right=df2,
    on='common_column',          # Column to join on
    how='left'                   # Join type: left, right, inner, outer
)

# 2. Concat: Stack DataFrames vertically or horizontally
df_vertical = pd.concat(
    [df1, df2],
    axis=0,                      # 0 for vertical, 1 for horizontal
    ignore_index=True            # Reset index
)

# 3. Join: Combine based on index
df_joined = df1.join(
    df2,
    on='index_column',
    how='left'
)`}
        />
      </Paper>

      <Paper p="lg" radius="md" className="bg-slate-50">
        <Title order={2}>Common Integration Patterns</Title>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
          <div>
            <Title order={3} size="h5">When to Use Merge</Title>
            <List size="sm" mt="xs">
              <List.Item>Combining related data with shared keys</List.Item>
              <List.Item>Matching records from different sources</List.Item>
              <List.Item>Need control over how records are matched</List.Item>
            </List>
          </div>
          <div>
            <Title order={3} size="h5">When to Use Concat</Title>
            <List size="sm" mt="xs">
              <List.Item>Combining similar data structures</List.Item>
              <List.Item>Appending new records to existing data</List.Item>
              <List.Item>Combining data with the same columns</List.Item>
            </List>
          </div>
        </div>
      </Paper>
    </Stack>
  );
};

export default ManipulatingSources;