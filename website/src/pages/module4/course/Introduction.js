import React from "react";
import { Container, Title, Text, Grid, Card, List, Box, Accordion, Image } from '@mantine/core';
import { IconDatabase, IconFileText, IconServer, IconRss, IconWorld, IconDatabaseImport, IconChartBar, IconCloudDataConnection } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";

const Introduction = () => {
  return (
    <Container fluid>
      <Title order={1} my="md">Introduction to Data Collection</Title>

      <section>
        <Title order={2} id="data-pipeline">Data Pipeline Process</Title>
        <Text my="md">
          Data collection is the first critical step in the data pipeline, where you gather raw data from various sources to feed into your data analysis, model building, or reporting.
        </Text>
        <Box ta="center" mb="xl">
          <Image
            src="/api/placeholder/800/200"
            alt="Data Pipeline Process"
            caption="The Data Pipeline Process"
          />
        </Box>
      </section>

      <section>
        <Title order={2} id="types">Types of Data</Title>
        <Text>Data can be categorized into three main types based on its structure:</Text>
        
        <Grid my="md">
          <Grid.Col span={4}>
            <Card shadow="sm" padding="lg">
              <Title order={3}>Structured Data</Title>
              <Text>
                Organized in a predefined format, typically in tables with rows and columns.
              </Text>
              <List size="sm">
                <List.Item>Examples: Relational databases, spreadsheets</List.Item>
                <List.Item>Easy to search and analyze</List.Item>
              </List>
            </Card>
          </Grid.Col>
          <Grid.Col span={4}>
            <Card shadow="sm" padding="lg">
              <Title order={3}>Semi-structured Data</Title>
              <Text>
                Has some organizational properties but doesn't conform to a rigid structure.
              </Text>
              <List size="sm">
                <List.Item>Examples: JSON, XML</List.Item>
                <List.Item>More flexible than structured data</List.Item>
              </List>
            </Card>
          </Grid.Col>
          <Grid.Col span={4}>
            <Card shadow="sm" padding="lg">
              <Title order={3}>Unstructured Data</Title>
              <Text>
                Data with no predefined format or organization.
              </Text>
              <List size="sm">
                <List.Item>Examples: Text documents, images, audio files</List.Item>
                <List.Item>Most abundant form of data</List.Item>
              </List>
            </Card>
          </Grid.Col>
        </Grid>
      </section>

      <section>
        <Title order={2} id="sources">Data Sources</Title>
        <Text>Data can be collected from various sources, including:</Text>
        <Grid my="md">
          {[
            { Icon: IconFileText, text: "Files\n(CSV, Excel, etc.)" },
            { Icon: IconDatabase, text: "Databases\n(SQL, NoSQL)" },
            { Icon: IconServer, text: "APIs\n(web services)" },
            { Icon: IconWorld, text: "Web scraping" }
          ].map(({ Icon, text }, index) => (
            <Grid.Col span={2} key={index}>
              <Box ta="center">
                <Icon size={48} />
                <Text size="sm" mt="xs">{text}</Text>
              </Box>
            </Grid.Col>
          ))}
        </Grid>
      </section>

      <section>
  <Title order={2} id="batch-streaming">Batch vs. Streaming Data Collection</Title>
  
  <Title order={3}>Batch Data Collection</Title>
  <Text>
    Batch data collection involves gathering and processing data in large, discrete chunks at scheduled intervals.
  </Text>
  <List>
    <List.Item>Suitable for large volumes of historical data</List.Item>
    <List.Item>Processed periodically (e.g., daily, weekly)</List.Item>
    <List.Item>Often used for complex analytics and reporting</List.Item>
  </List>

  <Title order={3}>Streaming Data Collection</Title>
  <Text>
    Streaming data collection involves gathering and processing data in real-time or near real-time as it's generated.
  </Text>
  <List>
    <List.Item>Suitable for real-time analytics and monitoring</List.Item>
    <List.Item>Processed continuously as data arrives</List.Item>
    <List.Item>Often used for time-sensitive applications</List.Item>
  </List>
</section>


      <section>
        <Title order={2} id="data-size">Data Size</Title>
        <Text>
          The size of collected data can significantly impact processing methods:
        </Text>
        <List>
          <List.Item>Larger datasets may require advanced storage formats (e.g., Parquet)</List.Item>
          <List.Item>Big data might necessitate distributed systems (e.g., Hadoop)</List.Item>
          <List.Item>Data size influences choice of tools and processing techniques</List.Item>
        </List>
        <Box ta="center" my="md">
          <IconChartBar size={64} />
          <Text size="sm" mt="xs">Data Size Considerations</Text>
        </Box>
      </section>

      <section>
        <Title order={2} id="data-quality">Data Quality</Title>
        <Text>
          Ensuring clean, accurate, and consistent data is essential before merging and analyzing the data.
        </Text>
        <List>
          <List.Item>Completeness: Check for missing values</List.Item>
          <List.Item>Accuracy: Verify data against trusted sources</List.Item>
          <List.Item>Consistency: Ensure uniform data across sources</List.Item>
          <List.Item>Timeliness: Confirm data is up-to-date</List.Item>
        </List>
      </section>

      <section>
        <Title order={2} id="managing-sources">Managing Multiple Sources</Title>
        <Text>
          When working with diverse data sources, it's essential to manage multiple datasets effectively. This often requires the use of unique identifiers (IDs) to join and merge datasets efficiently.
        </Text>
        <Box ta="center" my="md">
          <IconCloudDataConnection size={64} />
          <Text size="sm" mt="xs">Connecting Multiple Data Sources</Text>
        </Box>
        <Text>
          Tools like pandas offer powerful methods to handle this task. Here's a simple example of merging data from two sources:
        </Text>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Load data from a CSV file
df1 = pd.read_csv('sales_data.csv')

# Load data from a SQL database
from sqlalchemy import create_engine
engine = create_engine('sqlite:///customer_data.db')
df2 = pd.read_sql('SELECT * FROM customers', engine)

# Merge the datasets on a common column
merged_df = pd.merge(df1, df2, on='customer_id', how='inner')

print(merged_df.head())
          `}
        />
        <Text>
          This example demonstrates how to combine sales data from a CSV file with customer information from a SQL database, using a common 'customer_id' to join the datasets.
        </Text>
      </section>
    </Container>
  );
};

export default Introduction;