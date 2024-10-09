import React from 'react';
import { Container, Title, Text, List, Table, Group, Stack, Box, Anchor } from '@mantine/core';
import { IconDatabase, IconApi, IconWorld } from '@tabler/icons-react';
import DataInteractionPanel from 'components/DataInteractionPanel';
import CodeBlock from 'components/CodeBlock';

const CaseStudy = () => {
  const trainDataUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/module4_course_train.zip";
  const testDataUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/StoreE_data.csv";
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/module4_requirements.txt";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/module4_course.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/module4_course.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module4/course/module4_course.ipynb";
  const metadata = {
    description: "This dataset contains sales information from multiple stores to predict sales for StoreE.",
    source: "Multiple store sales records",
    target: "number_sold",
    listData: [
      { name: "store", description: "Store identifier" },
      { name: "weight", description: "Weight of the item" },
      { name: "length", description: "Length of the item" },
      { name: "width", description: "Width of the item" },
      { name: "days_since_last_sale", description: "Days since the last sale of the item" },
      { name: "days_in_stock", description: "Number of days the item has been in stock" },
      { name: "price", description: "Price of the item" },
      { name: "number_sold", description: "Number of items sold (target variable)" },
      { name: "last_updated", description: "Date of last update" },
      { name: "volume", description: "Volume of the item" },
    ],
  };

  return (
    <Container fluid py="xl">
      <Title order={1} mb="md">Course Data Collection Case Study</Title>

      <Box w="100%">
        <Section title="Objective">
          <Text>
            We want to predict the sales of <Text span fw={700}>StoreE</Text> based on the sales of other stores. We need to aggregate their data.
          </Text>
        </Section>

        <Section title="Data Sources">
          <List
            spacing="sm"
            icon={<IconDatabase size={16} />}
            mb="md"
          >
            <List.Item>5 stores (StoreA, StoreB, StoreC, StoreD, StoreE) each provide their relative data files in different formats (CSV, XLSX, JSON).</List.Item>
            <List.Item>
              Volume information is centralized for all stores and accessible through an API. 
              <Anchor href="https://www.raphaelcousin.com/module4/api-doc" target="_blank" ml={5} fw={500}>
                See API documentation
              </Anchor>
            </List.Item>
            <List.Item>
              Rating and number of reviews are displayed on a 
              <Anchor href="https://www.raphaelcousin.com/module4/scrapable-data" target="_blank" ml={5} fw={500}>
                web page
              </Anchor> 
              and need to be scraped.
            </List.Item>
          </List>
        </Section>

        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          requirementsUrl={requirementsUrl}
          metadata={metadata}
        />
      </Box>
    </Container>
  );
};

const Section = ({ title, children }) => (
  <Stack spacing="sm">
    <Title order={2}>{title}</Title>
    {children}
  </Stack>
);

export default CaseStudy;