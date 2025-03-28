import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
import DataInteractionPanel from "components/DataInteractionPanel";
import { Text } from '@mantine/core';
const ExerciseGenerativeModels = () => {
  const exerciseLinks = [
    // Add links to other exercises as needed
  ];

  const location = useLocation();
  const module = 11;

  const notebookUrl = process.env.PUBLIC_URL + "/modules/module11/TP_gen.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module11/TP_gen.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "/website/public/modules/module11/TP_gen.ipynb";

  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 11: Exercise Generative Models"
      courseLinks={exerciseLinks}
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

export default ExerciseGenerativeModels;