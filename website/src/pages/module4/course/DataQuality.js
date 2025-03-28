import React from 'react';
import { Text, Title, Paper, List, ThemeIcon } from '@mantine/core';
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
    <div>
      <Title order={1}>Data Quality</Title>
      
      <Text mt="md">
        Data quality is fundamental to reliable analysis and decision-making. A robust data quality framework 
        ensures that data is trustworthy, consistent, and fit for its intended use.
      </Text>

      <div className="grid grid-cols-1 gap-4 mt-8">
        {qualityDimensions.map((dim, index) => (
          <QualitySection key={index} {...dim} />
        ))}
      </div>

      <Paper p="lg" mt="xl" radius="md">
        <Title order={2} size="h3">Implementing Quality Controls</Title>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
          <div>
            <Title order={3} size="h5">Prevention Strategies</Title>
            <List size="sm" mt="xs">
              <List.Item>Standardized data collection procedures</List.Item>
              <List.Item>Clear data ownership and responsibilities</List.Item>
              <List.Item>Regular team training on data standards</List.Item>
              <List.Item>Automated validation at data entry</List.Item>
            </List>
          </div>
          <div>
            <Title order={3} size="h5">Monitoring Practices</Title>
            <List size="sm" mt="xs">
              <List.Item>Regular data quality assessments</List.Item>
              <List.Item>Quality metrics tracking and reporting</List.Item>
              <List.Item>Stakeholder feedback integration</List.Item>
              <List.Item>Documentation of quality issues</List.Item>
            </List>
          </div>
        </div>
      </Paper>
    </div>
  );
};

export default DataQuality;