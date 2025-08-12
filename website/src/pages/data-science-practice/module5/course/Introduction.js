import React from 'react';
import { Container, Title, Text, List, ThemeIcon, Space } from '@mantine/core';
import { IconCheckbox } from '@tabler/icons-react';

const Introduction = () => {
  return (
    <Container fluid>
      <Title order={1} id="introduction" mb="md">Introduction</Title>
      
      <Text mb="md">
        Data preprocessing is a crucial step in the data science pipeline. It involves transforming raw data into a clean and meaningful format. Proper preprocessing ensures that your data is consistent, complete, and optimized for machine learning algorithms.
      </Text>

      <Title order={2} id="key-steps" mb="sm">Key Steps in Data Preprocessing</Title>

      <List
        spacing="sm"
        size="sm"
        center
        icon={
          <ThemeIcon color="blue" size={24} radius="xl">
            <IconCheckbox size="1rem" />
          </ThemeIcon>
        }
      >
        <List.Item><strong>Handle Inconsistencies:</strong> Identify and correct data inconsistencies, such as different formats or units, to ensure data uniformity.</List.Item>
        <List.Item><strong>Handle Missing Values:</strong> Detect and address missing data through techniques like imputation or removal to maintain data integrity.</List.Item>
        <List.Item><strong>Handle Categorical Values:</strong> Convert categorical data into a format suitable for machine learning algorithms, often through encoding techniques.</List.Item>
        <List.Item><strong>Handle Duplicates:</strong> Identify and remove duplicate records to prevent bias and improve analysis accuracy.</List.Item>
        <List.Item><strong>Handle Outliers:</strong> Detect and manage extreme values that could skew your analysis or model performance.</List.Item>
        <List.Item><strong>Feature Engineering:</strong> Create new features or modify existing ones to better represent the underlying patterns in your data.</List.Item>
        <List.Item><strong>Scaling and Normalization:</strong> Adjust the scale of your features to ensure they contribute equally and improve model performance.</List.Item>
        <List.Item><strong>Feature Selection and Dimensionality Reduction:</strong> Choose the most relevant features and reduce data complexity to improve model efficiency and prevent overfitting.</List.Item>
      </List>

      <Space h="md" />
    </Container>
  );
};

export default Introduction;