import React, { lazy } from "react";
import { useLocation } from "react-router-dom";
import { Container, Text, Stack } from '@mantine/core';
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

const CourseDataCollection = () => {
  // const courseLinks = []
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction",
      component: lazy(() => import("pages/module4/course/Introduction")),
      subLinks: [
        { id: "types", label: "Types of Data" },
        { id: "sources", label: "Data Sources" },
        { id: "collection-methods", label: "Collection Methods" },
        { id: "integration-example", label: "Data Integration Example" },
        { id: "considerations", label: "Key Considerations" }
      ],
    },

    {
      to: "/files",
      label: "Files",
      component: lazy(() => import("pages/module4/course/Files")),
      subLinks: [
        { id: "metadata", label: "File Metadata" },
        { id: "structured", label: "Structured Files" },
        { id: "semi", label: "Semi-Structured Files" },
        { id: "unstructured", label: "Unstructured Files" },
      ],
    },
    {
      to: "/databases",
      label: "Databases",
      component: lazy(() => import("pages/module4/course/Databases")),
      subLinks: [
        { id: "sql-databases", label: "SQL Databases" },
        { id: "nosql-databases", label: "NoSQL Databases" },
      ],
    },
    {
      to: "/apis",
      label: "APIs",
      component: lazy(() => import("pages/module4/course/APIs")),
      subLinks: [
        { id: "api-overview", label: "High-level Overview" },
        { id: "authentication", label: "Authentication" },
        { id: "requests-responses", label: "Requests and Responses" },
        { id: "using-requests", label: "Using Python's requests Library" },
        { id: "rate-limiting", label: "Rate Limiting and Efficient Handling" },
      ],
    },
    {
      to: "/web-scraping",
      label: "Web Scraping",
      component: lazy(() => import("pages/module4/course/WebScraping")),
      subLinks: [
        { id: "scraping-overview", label: "High-level Overview" },
        { id: "ethical-considerations", label: "Ethical Considerations" },
        { id: "extracting-data", label: "Extracting Data from HTML Pages" },
        { id: "scraping-examples", label: "Simple Scraping Examples" },
      ],
    },
    {
      to: "/batch-vs-streaming",
      label: "Batch vs. Streaming Data Collection",
      component: lazy(() => import("pages/module4/course/BatchVsStreaming")),
      subLinks: [
        { id: "batch-collection", label: "Batch Collection" },
        { id: "streaming-collection", label: "Streaming/Event-based Collection" },
      ],
    },
    {
      to: "/data-quality",
      label: "Data Quality",
      component: lazy(() => import("pages/module4/course/DataQuality")),
      subLinks: [
        { id: "quality-overview", label: "Overview" },
        { id: "key-aspects", label: "Key Aspects of Data Quality" }
      ],
    },
    {
      to: "/manipulating-sources",
      label: "Manipulating Different Sources with pandas",
      component: lazy(() => import("pages/module4/course/ManipulatingSources")),
      subLinks: [
        { id: "combining-data", label: "Combining Data from Multiple Sources" },
        { id: "id-management", label: "ID Management" }
      ],
    },
    {
      to: "/case-study",
      label: "Case Study",
      component: lazy(() => import("pages/module4/course/CaseStudy")),
    },
  ];

  const location = useLocation();
  const module = 4;

  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 4: Data Collection"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <Container>
          <Stack spacing="md">
            <Text size="sm" color="dimmed">
              Last Updated: {new Date().toISOString().split('T')[0]}
            </Text>
          </Stack>
        </Container>
      )}
      <DynamicRoutes routes={courseLinks} type="course" />
    </ModuleFrame>
  );
};

export default CourseDataCollection;