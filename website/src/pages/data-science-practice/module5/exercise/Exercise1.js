import React from "react";
import { Container, Title, Text, List, Alert, Code } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import DataInteractionPanel from "components/DataInteractionPanel";
import CodeBlock from "components/CodeBlock";


const Exercise1 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/module5/exercise/module5_exercise_train.csv";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/module5/exercise/module5_exercise_test.csv";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module5/exercise/module5_exercise.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module5/exercise/module5_exercise.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module5/exercise/module5_exercise.ipynb";

  const metadata = {
    description: "This dataset represents aggregated daily electricity consumption in a major office zone in the United States from January 2015 to December 2019.",
    source: "Electricity Consumption Records",
    target: "Electricity Demand (MWh)",
    listData: [
      { name: "Date", description: "The specific day of observation" },
      { name: "Temperature1, ..., Temperature10", description: "Daily average temperature of the 10 stations (°C)" },
      { name: "Humidity", description: "Average daily humidity (%)" },
      { name: "Wind Speed", description: "Average wind speed (m/s or km/h)" },
      { name: "Oil Brent Price Indicator", description: "Ordinal feature representing fluctuations in oil prices" },
      { name: "Weather Condition", description: "Overall weather for each day (e.g., 'Sunny', 'Cloudy', 'Rainy', 'Snowy')" },
      { name: "Electricity Demand", description: "Total daily electricity consumption in megawatt-hours (MWh)" },
    ],
  };

  return (
    <Container fluid>
      <Title order={1} mb="md">Exercise 1: Data Preprocessing for Electricity Consumption Prediction</Title>
      <Text>  
        In this exercise, you will perform data preprocessing on an electricity consumption dataset to prepare it for modeling.
      </Text>

      <Title order={2} mt="xl" mb="md" id="overview">Overview</Title>
      <List>
        <List.Item>Load the dataset and explore its structure and statistics.</List.Item>
        <List.Item>Perform data analysis to identify preprocessing needs.</List.Item>
        <List.Item>Apply appropriate preprocessing techniques.</List.Item>
        <List.Item>Evaluate the effectiveness of your preprocessing steps.</List.Item>
      </List>
      <Alert icon={<IconInfoCircle size="1rem" />} title="" color="blue" mt="md">
        <List>
          <List.Item>The offices in this zone utilize fuel-powered generators for heating.</List.Item>
          <List.Item>Most workers don't work during weekends.</List.Item>
        </List>
      </Alert>
      <Title order={2} mt="xl" mb="md" id="data-analysis">Data Analysis</Title>
      <List>
        <List.Item>Inconsistencies in data types or values</List.Item>
        <List.Item>Duplicates </List.Item>
        <List.Item>Missing values</List.Item>
        <List.Item>Categorical variables and their distributions</List.Item>
        <List.Item>Outliers in numerical features</List.Item>
        <List.Item>Potential for feature engineering</List.Item>
        <List.Item>Opportunities for feature selection or dimensionality reduction</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md" id="preprocessing-strategy">Data Preprocessing Evaluation Strategy</Title>
      <Text>
        Develop a strategy to evaluate the effectiveness of your preprocessing steps:
      </Text>
      <List>
        <List.Item>Start with simple transformations and gradually improve your baseline.</List.Item>
        <List.Item>Use cross-validation to assess the impact of preprocessing on model performance.</List.Item>
        <List.Item>Compare different preprocessing techniques</List.Item>
      </List>

      <Title order={2} mt="xl" mb="md" id="submission">Generating Submission File</Title>
      <Text>
        After preprocessing the data and training your model, generate a submission file with your predictions on <Code>X_test</Code>:
      </Text>
      <List>
        <List.Item>
          Create a CSV file named <Code>submission.csv</Code> with two columns:
          <List withPadding>
            <List.Item><Code>date</Code>: The date for each prediction</List.Item>
            <List.Item><Code>electricity_demand</Code>: The predicted electricity demand in MWh</List.Item>
          </List>
        </List.Item>
      </List>
      <CodeBlock
        language="python"
        code={`
# Example of generating a submission file
predictions = model.predict(X_test)
submission = pd.DataFrame({
    'date': test_data['Date'],
    'electricity_demand': predictions
})
submission.to_csv('submission.csv', index=False)
        `}
      />

      <Title order={2} mt="xl" mb="md" id="evaluation">Evaluation</Title>
      <Text>
        Your preprocessing steps and predictions will be evaluated based on the following criteria:
      </Text>
      <Text>
        The error threshold for this exercise is a Mean Squared Error (MSE) of 1000. Ensure your predictions are accurate enough to meet this threshold.
      </Text>

      <Title order={2} mt="xl" mb="md" id="submission-requirements">Submission Requirements</Title>
      <List>
        <List.Item>
          A Jupyter Notebook named <Code>exercise1.ipynb</Code> containing your data analysis, preprocessing steps, and model development.
        </List.Item>
        <List.Item>
          A CSV file named <Code>submission.csv</Code> with your predictions.
        </List.Item>
        <List.Item>
          Save both files in the <Code>module5</Code> directory under your username folder.
        </List.Item>
      </List>

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

export default Exercise1;