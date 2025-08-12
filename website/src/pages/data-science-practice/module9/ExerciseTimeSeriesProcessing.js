import React, { lazy } from "react";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";
import DataInteractionPanel from "components/DataInteractionPanel";
import { Text, Grid } from '@mantine/core';
const ExerciseTimeSeriesProcessing = () => {
  const exerciseLinks = [
    // Add links to other exercises as needed
  ];const module = 9;

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
      
      <Grid>
        <Grid.Col span={{ md: 11 }}>
          <DynamicRoutes routes={exerciseLinks} type="exercise" />
        </Grid.Col>
      </Grid>
    </ModuleFrame>
  );
};

export default ExerciseTimeSeriesProcessing;