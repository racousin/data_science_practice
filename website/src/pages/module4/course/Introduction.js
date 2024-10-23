import React from 'react';
import { Container, Title, Text, Stack, Group, Paper, List, Divider } from '@mantine/core';
import { IconFileText, IconDatabase, IconServer, IconWorld, IconChartBar, IconCloudDataConnection } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";

const DataTypeSection = ({ title, description, examples, extraInfo }) => (
  <Paper p="lg" radius="md" className="bg-slate-50">
    <Stack gap="xs">
      <Text size="lg" fw={500} className="text-blue-600">{title}</Text>
      <Text>{description}</Text>
      <List size="sm">
        <List.Item>Examples: {examples}</List.Item>
        {extraInfo && <List.Item>{extraInfo}</List.Item>}
      </List>
    </Stack>
  </Paper>
);

const DataSourceIcon = ({ Icon, label }) => (
  <Stack gap="xs" align="center" className="w-32">
    <Icon size={32} className="text-blue-600" />
    <Text size="sm" ta="center">{label}</Text>
  </Stack>
);

const Introduction = () => {
  return (
    <Container fluid>
      <Stack gap="xl">
        {/* Main Introduction */}
        <Stack gap="md">
          <Text size="lg" c="dimmed">Module 4: Data Collection</Text>
          <Title order={1} id="introduction" className="text-blue-700">Introduction</Title>
          <Text>
            Data collection is the first critical step in the data pipeline, where you gather raw data from various 
            sources to feed into your data analysis, model building, or reporting. Understanding different types of 
            data and collection methods is crucial for effective data science projects.
          </Text>
        </Stack>

        {/* Types of Data */}
        <Stack gap="md">
          <Title order={2} id="types">Types of Data</Title>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <DataTypeSection
              title="Structured Data"
              description="Organized in a predefined format, typically in tables with rows and columns."
              examples="Relational databases, spreadsheets"
              extraInfo="Easy to search and analyze"
            />
            <DataTypeSection
              title="Semi-structured Data"
              description="Has some organizational properties but doesn't conform to a rigid structure."
              examples="JSON, XML"
              extraInfo="More flexible than structured data"
            />
            <DataTypeSection
              title="Unstructured Data"
              description="Data with no predefined format or organization."
              examples="Text documents, images, audio files"
              extraInfo="Most abundant form of data"
            />
          </div>
        </Stack>

        {/* Data Sources */}
        <Stack gap="md">
          <Title order={2} id="sources">Data Sources</Title>
          <Group justify="center" gap="xl">
            <DataSourceIcon Icon={IconFileText} label="Files(CSV, Excel, etc.)" />
            <DataSourceIcon Icon={IconDatabase} label="Databases(SQL, NoSQL)" />
            <DataSourceIcon Icon={IconServer} label="APIs(Web Services)" />
            <DataSourceIcon Icon={IconWorld} label="Web Scraping" />
          </Group>
        </Stack>

        {/* Collection Methods */}
        <Stack gap="md">
          <Title order={2} id="collection-methods">Collection Methods</Title>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Paper p="lg" radius="md" className="bg-slate-50">
              <Stack gap="sm">
                <Text size="lg" fw={500} className="text-blue-600">Batch Collection</Text>
                <List>
                  <List.Item>Processes data in large, discrete chunks</List.Item>
                  <List.Item>Scheduled intervals (daily, weekly, monthly)</List.Item>
                  <List.Item>Suitable for historical data analysis</List.Item>
                  <List.Item>Efficient for large-scale processing</List.Item>
                </List>
              </Stack>
            </Paper>
            <Paper p="lg" radius="md" className="bg-slate-50">
              <Stack gap="sm">
                <Text size="lg" fw={500} className="text-blue-600">Streaming Collection</Text>
                <List>
                  <List.Item>Real-time or near real-time processing</List.Item>
                  <List.Item>Continuous data ingestion</List.Item>
                  <List.Item>Ideal for time-sensitive applications</List.Item>
                  <List.Item>Enables immediate insights</List.Item>
                </List>
              </Stack>
            </Paper>
          </div>
        </Stack>

        {/* Data Integration Example */}
        <Stack gap="md">
          <Title order={2} id="integration-example">Data Integration Example</Title>
          <Text>
            Here's a practical example of combining data from multiple sources using pandas:
          </Text>
          <CodeBlock
            language="python"
            code={`
import pandas as pd
from sqlalchemy import create_engine

# Load sales data from CSV
sales_df = pd.read_csv('sales.csv')

# Load customer data from SQL database
engine = create_engine('sqlite:///customers.db')
customers_df = pd.read_sql('SELECT * FROM customers', engine)

# Merge datasets on customer_id
merged_df = pd.merge(
    sales_df, 
    customers_df,
    on='customer_id',
    how='inner'
)

# Basic data quality checks
print("Missing values:", merged_df.isnull().sum())
print("Duplicate rows:", merged_df.duplicated().sum())
            `}
          />
        </Stack>

        {/* Key Considerations */}
        <Stack gap="md">
          <Title order={2} id="considerations">Key Considerations</Title>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Paper p="lg" radius="md" className="bg-slate-50">
              <Stack gap="sm">
                <Group align="center" gap="xs">
                  <IconChartBar className="text-blue-600" />
                  <Text fw={500}>Data Size Impact</Text>
                </Group>
                <List>
                  <List.Item>Consider storage format (CSV vs Parquet)</List.Item>
                  <List.Item>Memory constraints and optimization</List.Item>
                  <List.Item>Processing time requirements</List.Item>
                </List>
              </Stack>
            </Paper>
            <Paper p="lg" radius="md" className="bg-slate-50">
              <Stack gap="sm">
                <Group align="center" gap="xs">
                  <IconCloudDataConnection className="text-blue-600" />
                  <Text fw={500}>Data Quality</Text>
                </Group>
                <List>
                  <List.Item>Completeness and accuracy checks</List.Item>
                  <List.Item>Consistency across sources</List.Item>
                  <List.Item>Regular validation procedures</List.Item>
                </List>
              </Stack>
            </Paper>
          </div>
        </Stack>
      </Stack>
    </Container>
  );
};

export default Introduction;