import React from 'react';
import { Text, Title, Paper, List, ThemeIcon, Flex, Image, Container, Grid } from '@mantine/core';
import { CheckCircle, AlertTriangle, RefreshCw, Filter } from 'lucide-react';

const QualitySection = ({ icon: Icon, title, description, points }) => (
  <Paper p="lg" radius="md" className="bg-slate-50">
    <div className="flex items-start gap-3">
      <ThemeIcon size={24} radius="md" className="mt-1">
        <Icon size={16} />
      </ThemeIcon>
      <div>
        <Title order={3} size="h4">{title}</Title>
        <Text size="sm" mt="xs">{description}</Text>
        <List size="sm" mt="md">
          {points.map((point, index) => (
            <List.Item key={index}>{point}</List.Item>
          ))}
        </List>
      </div>
    </div>
  </Paper>
);

const DataQuality = () => {
  const qualityDimensions = [
    {
      icon: CheckCircle,
      title: "Completeness",
      description: "Ensures all required data is present and accounted for",
      points: [
        "All mandatory fields are populated",
        "Coverage across time periods is consistent",
        "Sample sizes are statistically significant",
        "Missing data is documented and justified"
      ]
    },
    {
      icon: RefreshCw,
      title: "Consistency",
      description: "Data maintains uniformity across all sources and over time",
      points: [
        "Standardized formats across datasets",
        "Consistent naming conventions",
        "Uniform units of measurement",
        "Coherent relationships between fields"
      ]
    },
    {
      icon: AlertTriangle,
      title: "Accuracy",
      description: "Data correctly represents the real-world values it's meant to describe",
      points: [
        "Values are within expected ranges",
        "Data matches source systems",
        "Regular validation against known facts",
        "Error rates are monitored and documented"
      ]
    },
    {
      icon: Filter,
      title: "Validity",
      description: "Data conforms to defined business rules and formats",
      points: [
        "Data types are appropriate",
        "Values follow specified patterns",
        "Relationships between fields are logical",
        "Business rules are enforced"
      ]
    }
  ];

  return (
    <Container fluid>
      <div data-slide>
        <Title order={1}>Data Quality</Title>
        <Text size="lg" mt="md">
          Data quality is fundamental to reliable analysis and decision-making. A robust data quality framework
          ensures that data is trustworthy, consistent, and fit for its intended use.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Data Quality Framework</Title>
        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module4/quality.webp"
            alt="Data Quality Framework Visualization"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
          />
        </Flex>
        <Text>
          Data quality can be measured across four key dimensions, each addressing different aspects
          of data reliability and usability.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Four Dimensions of Data Quality</Title>
        <Grid gutter="lg">
          {qualityDimensions.map((dim, index) => (
            <Grid.Col span={6} key={index}>
              <QualitySection {...dim} />
            </Grid.Col>
          ))}
        </Grid>
      </div>
    </Container>
  );
};

export default DataQuality;