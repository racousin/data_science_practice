import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
import DataInteractionPanel from "components/DataInteractionPanel";
import { Text } from '@mantine/core';
const ExerciseTimeSeriesProcessing = () => {
  const exerciseLinks = [
    // Add links to other exercises as needed
  ];

  const location = useLocation();
  const module = 9;

  const notebookUrl = process.env.PUBLIC_URL + "/modules/module9/TP_ts.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module9/TP_ts.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "/website/public/modules/module9/TP_ts.ipynb";

  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 9: TimeSeries Processing"
    >
                    <Text mt="md" c="dimmed" size="sm">
          Author: Alessandro Bucci
        </Text>
      {location.pathname === `/module${module}/exercise` && (
        <>
          <DataInteractionPanel
            notebookUrl={notebookUrl}
            notebookHtmlUrl={notebookHtmlUrl}
            notebookColabUrl={notebookColabUrl}
          />
        </>
      )}
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={exerciseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default ExerciseTimeSeriesProcessing;