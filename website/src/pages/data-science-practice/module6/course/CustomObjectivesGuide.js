import React from 'react';
import { Container, Title, Text, Stack, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const CustomObjectivesGuide = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Custom Objectives Guide</Title>

      <Stack spacing="xl">
        <div data-slide>
          <Title order={2}>XGBoost: Full Customization</Title>

          <Text mb="md">
            XGBoost allows complete control over the optimization objective. You provide gradient and hessian (second derivative).
          </Text>

          <Title order={3} mt="lg">Asymmetric Loss (Penalize Underestimation)</Title>
          <CodeBlock language="python" code={`def custom_asymmetric_mse(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2 * residual, -0.5 * residual)
    hess = np.where(residual > 0, 2.0, 0.5)
    return grad, hess

model = xgb.XGBRegressor(objective=custom_asymmetric_mse)`} />

          <Title order={3} mt="lg">Focal Loss (Focus on Hard Examples)</Title>
          <CodeBlock language="python" code={`def focal_loss(y_true, y_pred, gamma=2.0):
    p = 1 / (1 + np.exp(-y_pred))  # Sigmoid
    grad = p - y_true
    hess = p * (1 - p) * ((1 - p + y_true * p) ** gamma)
    return grad * ((1 - p) ** gamma), hess`} />

          <Title order={3} mt="lg">Quantile Loss</Title>
          <CodeBlock language="python" code={`def quantile_loss(y_true, y_pred, alpha=0.9):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -alpha, -(alpha - 1))
    hess = np.ones_like(y_pred) * 1e-6  # Small constant
    return grad, hess`} />
        </div>

        <div data-slide>
          <Title order={2}>LightGBM: Custom Objectives</Title>

          <Text mb="md">
            LightGBM uses same format as XGBoost - provide gradient and hessian functions.
          </Text>

          <Title order={3} mt="lg">Huber Loss (Robust to Outliers)</Title>
          <CodeBlock language="python" code={`def huber_loss(y_true, y_pred, delta=1.0):
    residual = y_true - y_pred
    grad = np.where(np.abs(residual) <= delta,
                    -residual,
                    -delta * np.sign(residual))
    hess = np.where(np.abs(residual) <= delta, 1.0, 0.0)
    return grad, hess

lgb.LGBMRegressor(objective=huber_loss)`} />

          <Title order={3} mt="lg">Custom Evaluation Metric</Title>
          <CodeBlock language="python" code={`def custom_eval(y_true, y_pred):
    # Return (eval_name, eval_result, is_higher_better)
    error = np.mean(np.abs(y_true - y_pred) / (y_true + 1))
    return 'mape', error, False

model.fit(X, y, eval_metric=custom_eval)`} />
        </div>

        <div data-slide>
          <Title order={2}>CatBoost: Loss Functions</Title>

          <Text mb="md">
            CatBoost provides many built-in losses but also allows custom objectives via Python or C++.
          </Text>

          <Title order={3} mt="lg">Built-in Asymmetric Losses</Title>
          <CodeBlock language="python" code={`# Quantile regression
model = CatBoostRegressor(loss_function='Quantile:alpha=0.9')

# Asymmetric MSE with different weights
model = CatBoostRegressor(loss_function='RMSE',
                         class_weights=[0.5, 2.0])  # Penalize class 1 more`} />

          <Title order={3} mt="lg">Custom Python Objective</Title>
          <CodeBlock language="python" code={`class CustomObjective:
    def calc_ders_range(self, approxes, targets, weights):
        # approxes: predictions, targets: true values
        grad = targets - approxes  # First derivative
        hess = np.ones_like(targets)  # Second derivative
        return list(zip(grad, hess))

CatBoostRegressor(loss_function=CustomObjective())`} />
        </div>

        <div data-slide>
          <Title order={2}>Scikit-learn: Sample & Class Weights</Title>

          <Text mb="md">
            Most sklearn models don't allow custom losses but support sample/class weights to adjust importance.
          </Text>

          <Title order={3} mt="lg">Class Weights (Classification)</Title>
          <CodeBlock language="python" code={`# Automatic balancing
RandomForestClassifier(class_weight='balanced')

# Custom weights per class
LogisticRegression(class_weight={0: 1, 1: 10})  # 10x weight for class 1

# Compute balanced weights
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)`} />

          <Title order={3} mt="lg">Sample Weights (Any Task)</Title>
          <CodeBlock language="python" code={`# Weight by inverse frequency
sample_weights = 1 / np.bincount(y)[y]

# Weight by importance/confidence
weights = np.where(confidence > 0.8, 2.0, 1.0)

model.fit(X, y, sample_weight=weights)`} />

          <Title order={3} mt="lg">Custom Scorer for GridSearch</Title>
          <CodeBlock language="python" code={`from sklearn.metrics import make_scorer

def custom_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) ** 1.5)

scorer = make_scorer(custom_loss, greater_is_better=False)
GridSearchCV(model, params, scoring=scorer)`} />
        </div>

      </Stack>
    </Container>
  );
};

export default CustomObjectivesGuide;