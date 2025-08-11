import React from "react";
import { Container, Grid, Image } from '@mantine/core';
import { BlockMath, InlineMath } from "react-katex";
import CodeBlock from "components/CodeBlock";
const EvaluationMetrics = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Evaluation Metrics</h1>
      <p>
        Evaluation metrics quantify the difference between the target variable{" "}
        <InlineMath math="y" /> and the model's predictions{" "}
        <InlineMath math="\hat{y}" />, providing a measure of how well our model
        is performing. These metrics should reflect the specific problem and
        objectives of our project. It's crucial to monitor these metrics over
        time to assess if we are improving and to guide further model
        refinements.
      </p>
      <h2 id="regression-metrics">Regression Metrics</h2>
      <p>
        Regression problems involve predicting continuous numerical values. Key
        metrics include:
      </p>
      <h3>Mean Squared Error (MSE)</h3>
      <p>
        MSE measures the average squared difference between predicted and actual
        values. It's sensitive to outliers.
      </p>
      <BlockMath math="MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
        `}
      />
      <h3>Mean Absolute Error (MAE)</h3>
      <p>
        MAE measures the average absolute difference between predicted and
        actual values. It's more robust to outliers than MSE/RMSE.
      </p>
      <BlockMath math="MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
        `}
      />
      <h2 id="binary-classification-metrics">Binary Classification Metrics</h2>
      <p>
        Binary classification involves predicting one of two possible outcomes.
        Key metrics include:
      </p>
      <h3>Accuracy</h3>
      <p>
        Accuracy is the ratio of correct predictions to total predictions. It
        can be misleading for imbalanced datasets.
      </p>
      <BlockMath math="Accuracy = \frac{TP + TN}{TP + TN + FP + FN}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
        `}
      />
      <h3>Precision</h3>
      <p>
        Precision is the ratio of true positive predictions to total positive
        predictions. It's important when the cost of false positives is high.
      </p>
      <BlockMath math="Precision = \frac{TP}{TP + FP}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)
        `}
      />
      <h3>Recall (Sensitivity)</h3>
      <p>
        Recall is the ratio of true positive predictions to total actual
        positives. It's crucial when the cost of false negatives is high.
      </p>
      <BlockMath math="Recall = \frac{TP}{TP + FN}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)
        `}
      />
      <h3>F1 Score</h3>
      <p>
        F1 Score is the harmonic mean of precision and recall, providing a
        single score that balances both metrics.
      </p>
      <BlockMath math="F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
        `}
      />
      <h3>ROC AUC (Receiver Operating Characteristic Area Under Curve)</h3>
      <p>
        ROC AUC represents the model's ability to distinguish between classes.
        It's scale-invariant and classification-threshold-invariant.
      </p>
      {/* TODO explain the value of ROC auc */}
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_true, y_pred_proba)
        `}
      />
      <Grid className="justify-content-center">
        <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
          <div className="text-center">
            <Image src="/assets/module3/roc_auc.png" alt="roc_auc" fluid />
            <p>ROC AUC Curve</p>
          </div>
        </Grid.Col>
      </Grid>
      <h2 id="multi-class-classification-metrics">Multi-class Classification Metrics</h2>
        <p>
          Multi-class classification involves predicting one of three or more possible outcomes. Key metrics include:
        </p>
        <h3>Macro-averaged F1 Score</h3>
        <p>
          Calculates F1 score for each class independently and takes the unweighted mean.
          Gives equal importance to all classes, regardless of their frequency.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.metrics import f1_score
macro_f1 = f1_score(y_true, y_pred, average='macro')
      `}
      />
      <h3>Weighted-averaged F1 Score</h3>
        <p>
          Calculates F1 score for each class and takes the weighted mean based on class frequency.
          Accounts for class imbalance.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.metrics import f1_score
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
      `}
      />
      <h2 id="ranking-metrics">Ranking Metrics</h2>
      <p>
        Ranking problems involve ordering items based on relevance or
        importance. Key metrics include:
      </p>
      <h3>Mean Average Precision (MAP)</h3>
      <p>
        MAP measures the quality of rankings across multiple queries. It's
        sensitive to the entire ranking.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import average_precision_score
map_score = average_precision_score(y_true, y_score)
        `}
      />
      <h3>Normalized Discounted Cumulative Gain (NDCG)</h3>
      <p>
        NDCG measures the quality of rankings with emphasis on top-ranked items.
        It's useful for graded relevance scores and when order matters.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import ndcg_score
ndcg = ndcg_score(y_true, y_score)
        `}
      />
      <h2 id="time-series-metrics">Time Series Metrics</h2>
      <p>
        Time series forecasting involves predicting future values based on
        historical data. A key metric is:
      </p>
      <h3>Mean Absolute Percentage Error (MAPE)</h3>
      <p>
        MAPE measures the average percentage difference between predicted and
        actual values. It's scale-independent, useful for comparing forecasts
        across different scales.
      </p>
      <BlockMath math="MAPE = \frac{1}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right| \cdot 100" />
      <CodeBlock
        language="python"
        code={`
import numpy as np
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape_score = mape(y_true, y_pred)
        `}
      />
      <h2 id="choosing-metrics">Choosing the Right Metric</h2>
      <p>When selecting evaluation metrics, consider:</p>
      <ul>
        <li>Nature of the problem (regression, classification, ranking, etc.)</li>
        <li>Business objectives and cost of errors</li>
        <li>Class balance in classification problems</li>
        <li>Sensitivity to outliers</li>
      </ul>
      <p>
        Remember that using multiple metrics often provides a more comprehensive
        view of model performance. Always interpret metrics in the context of
        your specific problem and goals, and monitor them over time to track
        improvements.
      </p>
    </Container>
  );
};
export default EvaluationMetrics;