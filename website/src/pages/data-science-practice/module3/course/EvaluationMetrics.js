import React from "react";
import { Container, Grid, Image, Title, Text, List, Flex } from '@mantine/core';
import { BlockMath, InlineMath } from "react-katex";
import CodeBlock from "components/CodeBlock";
const EvaluationMetrics = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={1} mb="md">Evaluation Metrics</Title>
        <Text size="md" mb="md">
          Evaluation metrics quantify the difference between the target variable{" "}
          <InlineMath math="y" /> and the model's predictions{" "}
          <InlineMath math="\hat{y}" />, providing a measure of how well our model
          is performing. These metrics should reflect the specific problem and
          objectives of our project. It's crucial to monitor these metrics over
          time to assess if we are improving and to guide further model
          refinements.
        </Text>

        <Text size="lg" mb="md">
          Metrics functions quantify the error/distance between real targets <InlineMath>{`y`}</InlineMath> and our model outputs <InlineMath>{`\\hat{y} = f_{\\theta}(x)`}</InlineMath>.
        </Text>   
      </div>

      <div data-slide>
        <Title order={2} mb="md">Regression Metrics</Title>
        <Text size="md" mb="md">
          Regression problems involve predicting continuous numerical values. Key
          metrics include:
        </Text>
        <Title order={3} mb="md">Mean Squared Error (MSE)</Title>
        <Text size="md" mb="md">
          MSE measures the average squared difference between predicted and actual
          values. It's sensitive to outliers.
        </Text>
        <BlockMath math="MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2" />
        <CodeBlock
          language="python"
          code={`from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)`}
        />
        <Title order={3} mb="md">Mean Absolute Error (MAE)</Title>
        <Text size="md" mb="md">
          MAE measures the average absolute difference between predicted and
          actual values. It's more robust to outliers than MSE/RMSE.
        </Text>
        <BlockMath math="MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|" />
        <CodeBlock
          language="python"
          code={`from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)`}
        />
      </div>
<div data-slide>
                            <Flex direction="column" align="center">
                      <Image
                        src="/assets/data-science-practice/module3/regression.png"
                        alt="Yutong Liu & The Bigger Picture"
                        style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
</div>

      <div data-slide>
        <Title order={3} mb="md">MSE vs MAE: Sensitivity to Large Errors</Title>

        <Title order={4} mb="md">Example: Impact of Outliers</Title>
        <Text size="md" mb="md">
          Consider three predictions with errors: 1, 2, and 10 units.
        </Text>

        <Text size="md" mb="md">
          For MAE: <InlineMath math="MAE = \frac{|1| + |2| + |10|}{3} = \frac{13}{3} = 4.33" />
        </Text>

        <Text size="md" mb="md">
          For MSE: <InlineMath math="MSE = \frac{1^2 + 2^2 + 10^2}{3} = \frac{1 + 4 + 100}{3} = 35" />
        </Text>

        <Text size="md" mb="md">
          The large error (10) contributes <InlineMath math="\frac{100}{105} = 95.2\%" /> to MSE but only <InlineMath math="\frac{10}{13} = 76.9\%" /> to MAE. This makes MSE more sensitive to outliers and large errors, while MAE provides a more robust measure when outliers are present.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Binary Classification Metrics</Title>
        <Text size="md" mb="md">
          Binary classification involves predicting one of two possible outcomes.
          Key metrics include:
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Accuracy</Title>
        <Text size="md" mb="md">
          Accuracy is the ratio of correct predictions to total predictions. It
          can be misleading for imbalanced datasets.
        </Text>
        <BlockMath math="Accuracy = \frac{TP + TN}{TP + TN + FP + FN}" />
        <CodeBlock
          language="python"
          code={`from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="md">Precision</Title>
        <Text size="md" mb="md">
          Precision is the ratio of true positive predictions to total positive
          predictions. It's important when the cost of false positives is high.
        </Text>
        <BlockMath math="Precision = \frac{TP}{TP + FP}" />
        <CodeBlock
          language="python"
          code={`from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)`}
        />
        <Title order={3} mb="md">Recall (Sensitivity)</Title>
        <Text size="md" mb="md">
          Recall is the ratio of true positive predictions to total actual
          positives. It's crucial when the cost of false negatives is high.
        </Text>
        <BlockMath math="Recall = \frac{TP}{TP + FN}" />
        <CodeBlock
          language="python"
          code={`from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="md">Classical Issues with Accuracy</Title>
        <Text size="md" mb="md">
          While accuracy is intuitive, it can be misleading in several common scenarios:
        </Text>

        <Title order={3} mb="md">Imbalanced Datasets</Title>
        <Text size="md" mb="md">
          When classes are heavily imbalanced, accuracy fails to capture true model performance.
        </Text>
        <Text size="md" mb="md">
          Consider an email spam detection dataset: 99% of emails are not spam, 1% are spam.
        </Text>
        <Text size="md" mb="md">
          A naive model that always predicts "not spam" achieves:
        </Text>
        <Text size="md" mb="md">
          Accuracy = <InlineMath math="\frac{99}{100} = 99\%" /> (misleading!)
        </Text>
        <Text size="md" mb="md">
          Recall = <InlineMath math="\frac{0}{1} = 0\%" /> (reveals the problem)
        </Text>
        <Text size="md" mb="md">
          The model appears excellent but catches zero spam emails.
        </Text>

        <Title order={3} mb="md">Medical Diagnosis Example</Title>
        <Text size="md" mb="md">
          In medical screening where disease prevalence is low, accuracy can hide critical failures.
        </Text>
        <Text size="md" mb="md">
          Consider a rare disease affecting 2% of the population. A model always predicting "healthy" achieves:
        </Text>
        <Text size="md" mb="md">
          Accuracy = <InlineMath math="\frac{98}{100} = 98\%" /> but misses 100% of actual cases - catastrophic failure!
        </Text>
        <Text size="md" mb="md">
          Better metrics for medical diagnosis:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item><strong>High Recall:</strong> Don't miss sick patients</List.Item>
          <List.Item><strong>Precision:</strong> Minimize false alarms</List.Item>
          <List.Item><strong>F1 Score:</strong> Balance both concerns</List.Item>
        </List>

      </div>

<div data-slide>
                            <Flex direction="column" align="center">
                      <Image
                        src="/assets/data-science-practice/module3/Precisionrecall.png"
                        alt="Yutong Liu & The Bigger Picture"
                        style={{ maxWidth: 'min(500px, 60vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
</div>
      <div data-slide>
        <Title order={3} mb="md">F1 Score</Title>
        <Text size="md" mb="md">
          F1 Score is the harmonic mean of precision and recall, providing a
          single score that balances both metrics.
        </Text>
        <BlockMath math="F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}" />
        <CodeBlock
          language="python"
          code={`from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="md">ROC AUC (Receiver Operating Characteristic Area Under Curve)</Title>
        <Text size="md" mb="md">
          ROC AUC measures the model's confidence and ability to distinguish between classes.
        </Text>

        <Text size="md" mb="md">
          The model outputs probabilities <InlineMath math="P(y=1|x) \in [0,1]" />, where:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>0 = completely confident the sample is negative class</List.Item>
          <List.Item>1 = completely confident the sample is positive class</List.Item>
          <List.Item>By default, we use threshold = 0.5 for final predictions</List.Item>
        </List>

        <Text size="md" mb="md">
          ROC AUC evaluates how well the model separates classes across all possible thresholds:
        </Text>
        <List spacing="sm" mb="md">
          <List.Item><strong>AUC = 1:</strong> Perfect separation (ideal confidence)</List.Item>
          <List.Item><strong>AUC = 0.5:</strong> Random guessing (no confidence)</List.Item>
          <List.Item><strong>AUC = 0:</strong> Perfect but inverted (confident but wrong)</List.Item>
        </List>

            <Flex direction="column" align="center">
              <Image
                 src="/assets/data-science-practice/module3/roc_auc.png"
                alt="Yutong Liu & The Bigger Picture"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
            </Flex>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Ranking Metrics</Title>
        <Text size="md" mb="md">
          Ranking problems involve ordering items based on relevance or
          importance. Key metrics include:
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Mean Average Precision (MAP)</Title>
        <Text size="md" mb="md">
          MAP measures the quality of rankings across multiple queries. It's
          sensitive to the entire ranking.
        </Text>
        <CodeBlock
          language="python"
          code={`from sklearn.metrics import average_precision_score
map_score = average_precision_score(y_true, y_score)`}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="md">Normalized Discounted Cumulative Gain (NDCG)</Title>
        <Text size="md" mb="md">
          NDCG measures the quality of rankings with emphasis on top-ranked items.
          It's useful for graded relevance scores and when order matters.
        </Text>
        <CodeBlock
          language="python"
          code={`from sklearn.metrics import ndcg_score
ndcg = ndcg_score(y_true, y_score)`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="md">Time Series Metrics</Title>
        <Text size="md" mb="md">
          Time series forecasting involves predicting future values based on
          historical data. A key metric is:
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Mean Absolute Percentage Error (MAPE)</Title>
        <Text size="md" mb="md">
          MAPE measures the average percentage difference between predicted and
          actual values. It's scale-independent, useful for comparing forecasts
          across different scales.
        </Text>
        <BlockMath math="MAPE = \frac{1}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right| \cdot 100" />
        <CodeBlock
          language="python"
          code={`import numpy as np

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_score = mape(y_true, y_pred)`}
        />
      </div>
    </Container>
  );
};
export default EvaluationMetrics;