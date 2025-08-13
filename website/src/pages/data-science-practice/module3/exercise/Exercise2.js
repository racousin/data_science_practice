import React from "react";
import { Container, Grid } from '@mantine/core';
import DataInteractionPanel from "components/DataInteractionPanel";
import CodeBlock from "components/CodeBlock";
const Exercise2 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module3/exercise/module3_exercise2_train.csv";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module3/exercise/module3_exercise2_test.csv";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module3/exercise/module3_exercise2.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module3/exercise/module3_exercise2.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module3/exercise/module3_exercise2.ipynb";
  const metadata = {
    description: "This dataset includes various health metrics crucial for predicting diabetes in Pima Indian women.",
    source: "National Institute of Diabetes and Digestive and Kidney Diseases",
    target: "Outcome",
    listData: [
      { name: "Pregnancies", description: "Number of times pregnant" },
      { name: "Glucose", description: "Plasma glucose concentration at 2 hours in an oral glucose tolerance test" },
      { name: "BloodPressure", description: "Diastolic blood pressure (mm Hg)" },
      { name: "SkinThickness", description: "Triceps skin fold thickness (mm)" },
      { name: "Insulin", description: "2-Hour serum insulin (mu U/ml)" },
      { name: "BMI", description: "Body mass index (weight in kg/(height in m)^2)" },
      { name: "DiabetesPedigreeFunction", description: "Diabetes pedigree function" },
      { name: "Age", description: "Age (years)" },
      { name: "Outcome", description: "Class variable (0 or 1, where 1 indicates tested positive for diabetes)" },
    ],
  };
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 2: Diabetes Prediction Model</h1>
      <p>
        In this exercise, you will develop a model to predict diabetes in Pima Indian women based on various health metrics.
      </p>
      <Grid>
        <Grid.Col>
          <h2>Overview</h2>
          <ul>
            <li>Load the Pima Indians Diabetes dataset and explore its structure and statistics.</li>
            <li>Perform necessary data preprocessing and feature engineering.</li>
            <li>Develop a model to predict the likelihood of diabetes.</li>
            <li>Evaluate the model's performance using appropriate metrics, with a focus on accuracy.</li>
            <li>Generate predictions for the test dataset.</li>
          </ul>
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <h2>Expected Output</h2>
          <p>
            You are expected to submit two files:
          </p>
          <ol>
            <li>A Jupyter Notebook named <code>exercise2.ipynb</code> containing your model development, evaluation, and analysis.</li>
            <li>A CSV file named <code>submission.csv</code> with your predictions.</li>
          </ol>
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <h2>Submission Requirements</h2>
          <ol>
            <li>
              The CSV file <code>submission.csv</code> should have two columns:
              <ul>
                <li><code>id</code>: The identifier for each prediction (row number in the test set).</li>
                <li><code>Outcome</code>: The predicted values (0 or 1).</li>
              </ul>
            </li>
            <CodeBlock
              code={`id,Outcome\n1,0\n2,1\n3,0\n...`}
            />
            <li>
              Save both the <code>exercise2.ipynb</code> notebook and the <code>submission.csv</code> file in the <code>module3</code> directory under your username folder.
            </li>
          </ol>
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <h2>Review PR Validation</h2>
          <p>Once you've completed your model development and created your submission, follow these steps:</p>
          <ol>
            <li>Push your changes to your fork of the repository.</li>
            <li>Create a Pull Request (PR) to merge your changes into the main branch.</li>
            <li>Your PR will trigger an automated validation process that checks:</li>
            <ul>
              <li>The presence of the <code>exercise2.ipynb</code> and <code>submission.csv</code> files in the correct location.</li>
              <li>The notebook's ability to run without errors.</li>
              <li>The format and content of the submission CSV file.</li>
              <li>The model's performance against the accuracy threshold.</li>
            </ul>
            <li>Once the automated checks pass, a reviewer will be assigned to provide feedback on your work.</li>
            <li>The reviewer will assess:</li>
            <ul>
              <li>The thoroughness of your data analysis and preprocessing.</li>
              <li>The appropriateness of your modeling approach.</li>
              <li>The quality of your code and documentation.</li>
              <li>Your interpretation of the results and any insights gained.</li>
            </ul>
            <li>Address any feedback or suggestions provided by the reviewer.</li>
            <li>Once approved, your PR will be merged, completing the exercise.</li>
          </ol>
          <p>
            <strong>Note:</strong> The validation threshold for this exercise is an accuracy above 80% on the test set. Ensure your model meets or exceeds this threshold.
          </p>
        </Grid.Col>
      </Grid>
      <Grid>
        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          metadata={metadata}
        />
      </Grid>
    </Container>
  );
};
export default Exercise2;