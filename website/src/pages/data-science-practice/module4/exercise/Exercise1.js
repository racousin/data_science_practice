import React from 'react';
import { Container, Title, Text, List, Table, Group, Stack, Box, Anchor, Code } from '@mantine/core';
import { IconDatabase, IconApi, IconWorld } from '@tabler/icons-react';
import DataInteractionPanel from 'components/DataInteractionPanel';
import CodeBlock from 'components/CodeBlock';

const Exercise1 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/module4/exercise/module4_exercise_train.zip";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/module4/exercise/Neighborhood_Market_data.csv";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module4/exercise/module4_exercise1.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module4/exercise/module4_exercise1.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "/modules/module4/exercise/module4_exercise1.ipynb";

  const metadata = {
    description: "This dataset contains sales information from multiple stores to predict sales for Neighborhood_Market.",
    source: "Multiple store sales records",
    target: "quantity_sold",
    listData: [
      { name: "item_code", description: "Unique identifier for each item" },
      { name: "store_name", description: "Name of the store" },
      { name: "mass", description: "Mass of the item" },
      { name: "dimension_length", description: "Length of the item" },
      { name: "dimension_width", description: "Width of the item" },
      { name: "dimension_height", description: "Height of the item" },
      { name: "customer_score", description: "Customer rating of the item" },
      { name: "total_reviews", description: "Total number of customer reviews" },
      { name: "days_since_last_purchase", description: "Days since the item was last purchased" },
      { name: "package_volume", description: "Volume of the item's package" },
      { name: "stock_age", description: "Age of the item in stock" },
      { name: "unit_cost", description: "Cost per unit of the item" },
      { name: "quantity_sold", description: "Number of units sold (target variable)" },
      { name: "last_modified", description: "Date of last update to the record" },
    ],
  };

  return (
    <Container fluid py="xl">
      <Title order={1} mb="md">Exercise 1: Multi-Sources Sales Prediction</Title>

      <Stack spacing="xl">
        <Section title="Objective">
          <Text>
            In this exercise, you will develop a baseline model to predict the <Code>quantity sold </Code> for 
            <Text span fw={700}> Neighborhood_Market</Text> based on sales data from multiple stores.
          </Text>
        </Section>

        <Section title="Data Sources">
          <List
            spacing="sm"
            icon={<IconDatabase size={16} />}
            mb="md"
          >
            <List.Item>5 stores (CityMart, Greenfield_Grocers, SuperSaver_Outlet, HighStreet_Bazaar, Neighborhood_Market) each provide their data files in different formats (CSV, XLSX, JSON).</List.Item>
            <List.Item>
            <Code>Unit cost</Code> information is centralized for all stores and accessible through an API. 
              <Anchor href="https://www.raphaelcousin.com/module4/api-doc" target="_blank" ml={5} fw={500}>
                See API documentation
              </Anchor>
            </List.Item>
            <List.Item>
            <Code>Customer score</Code> and <Code>total reviews</Code> are displayed on a 
              <Anchor href="https://www.raphaelcousin.com/module4/scrapable-data" target="_blank" ml={5} fw={500}>
                web page
              </Anchor> 
              and need to be scraped.
            </List.Item>
          </List>
        </Section>

        <Section title="Task Overview">
          <List type="ordered">
            <List.Item>Load and combine data from multiple sources (files, API, web scraping).</List.Item>
            <List.Item>Perform exploratory data analysis (EDA) on the combined dataset.</List.Item>
            <List.Item>Prepare the data for modeling based on your EDA findings.</List.Item>
            <List.Item>Develop and evaluate a baseline model.</List.Item>
            <List.Item>Generate predictions for the test dataset Neighborhood_Market.</List.Item>
          </List>
        </Section>

        <Section title="Expected Output">
          <Text>You are expected to submit two files:</Text>
          <List>
            <List.Item>A Jupyter Notebook named <Code>exercise1.ipynb</Code> containing your data processing, model development, and evaluation.</List.Item>
            <List.Item>A CSV file named <Code>submission.csv</Code> with your predictions.</List.Item>
          </List>
        </Section>

        <Section title="Submission Requirements">
          <List type="ordered">
            <List.Item>
              A CSV file named <Code>submission.csv</Code> with two columns:
              <List withPadding listStyleType="disc">
                <List.Item><Code>item_code</Code>: The identifier for each item.</List.Item>
                <List.Item><Code>quantity_sold</Code>: The predicted values.</List.Item>
              </List>
            </List.Item>
            <List.Item>
              <CodeBlock
                language="text"
                code=
{`item_code,quantity_sold
A001,50
A002,75
A003,100
...`}
              />
            </List.Item>
            <List.Item>
              Save both the <Code>exercise1.ipynb</Code> notebook and the <Code>submission.csv</Code> file in the <Code>module4</Code> directory under your username folder.
            </List.Item>
          </List>
        </Section>

        <Section title="Evaluation">
          <Text>Your baseline model and predictions will be evaluated based on the Mean Absolute Error (MAE).</Text>
          <Text fw={700}>The error threshold for this exercise is an MAE of 20. Ensure your predictions are accurate enough to meet this threshold.</Text>
        </Section>
        </Stack>
        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          metadata={metadata}
        />

    </Container>
  );
};

const Section = ({ title, children }) => (
  <Stack spacing="sm">
    <Title order={2}>{title}</Title>
    {children}
  </Stack>
);

export default Exercise1;