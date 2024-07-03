import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { InlineMath, BlockMath } from "react-katex";
import CodeBlock from "components/CodeBlock";

const ModelEvaluationValidation = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Model Evaluation and Validation</h1>

      <section>
        <h2 id="importance">Importance of Model Evaluation</h2>
        <p>
          Model evaluation is a crucial step in the machine learning pipeline.
          It helps us understand how well our model performs on unseen data and
          whether it's ready for deployment. Proper evaluation ensures that our
          model generalizes well and isn't overfitting to the training data.
        </p>
      </section>

      <section>
        <h2 id="overfitting-underfitting">
          Understanding Overfitting and Underfitting
        </h2>
        <p>
          Overfitting and underfitting are common problems in machine learning:
        </p>
        <ul>
          <li>
            <strong>Overfitting:</strong> The model learns the training data too
            well, including its noise, leading to poor generalization.
          </li>
          <li>
            <strong>Underfitting:</strong> The model is too simple to capture
            the underlying patterns in the data.
          </li>
        </ul>

        <h3>Bias-Variance Tradeoff</h3>
        <p>
          The bias-variance tradeoff is a fundamental concept in machine
          learning:
        </p>
        <ul>
          <li>
            <strong>Bias:</strong> The error introduced by approximating a
            real-world problem with a simplified model.
          </li>
          <li>
            <strong>Variance:</strong> The amount by which the model would
            change if we estimated it using a different training dataset.
          </li>
        </ul>
        <p>
          High bias can lead to underfitting, while high variance can lead to
          overfitting. The goal is to find the right balance.
        </p>

        <h3>Learning Curves</h3>
        <p>
          Learning curves are a valuable tool for diagnosing bias and variance
          issues:
        </p>
        <CodeBlock
          language="python"
          code={`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt
          `}
        />
      </section>

      <section>
        <h2 id="train-test-splits">Train-Test-Validation Splits</h2>
        <p>
          Properly splitting your data is crucial for reliable model evaluation:
        </p>

        <h3>Holdout Method</h3>
        <p>
          The simplest approach is to split your data into training and test
          sets:
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
          `}
        />

        <h3>Cross-Validation Techniques</h3>
        <p>
          Cross-validation provides a more robust evaluation, especially for
          smaller datasets:
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

# Stratified K-Fold (for classification problems)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)

# Leave-One-Out Cross-Validation
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
          `}
        />

        <h3>Time Series Cross-Validation</h3>
        <p>For time series data, we need to respect the temporal order:</p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Fit and evaluate your model here
          `}
        />
      </section>

      <section>
        <h2 id="evaluation-metrics">Evaluation Metrics</h2>

        <h3>For Regression Problems</h3>
        <ul>
          <li>
            <strong>Mean Squared Error (MSE):</strong>
            <BlockMath math="MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2" />
          </li>
          <li>
            <strong>Root Mean Squared Error (RMSE):</strong>
            <BlockMath math="RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}" />
          </li>
          <li>
            <strong>Mean Absolute Error (MAE):</strong>
            <BlockMath math="MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|" />
          </li>
          <li>
            <strong>R-squared:</strong>
            <BlockMath math="R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}" />
          </li>
        </ul>

        <h3>For Classification Problems</h3>
        <ul>
          <li>
            <strong>Accuracy:</strong> Proportion of correct predictions.
          </li>
          <li>
            <strong>Precision:</strong> Proportion of true positives among
            positive predictions.
          </li>
          <li>
            <strong>Recall:</strong> Proportion of true positives among actual
            positives.
          </li>
          <li>
            <strong>F1-score:</strong> Harmonic mean of precision and recall.
          </li>
          <li>
            <strong>ROC curve and AUC:</strong> Plots the true positive rate
            against the false positive rate.
          </li>
        </ul>

        <h3>Confusion Matrix</h3>
        <p>
          A table showing the number of correct and incorrect predictions made
          by a classification model:
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
          `}
        />

        <h3>For Ranking Problems</h3>
        <ul>
          <li>
            <strong>Mean Average Precision (MAP):</strong> Average of the
            precision scores at each relevant item.
          </li>
          <li>
            <strong>Normalized Discounted Cumulative Gain (NDCG):</strong>{" "}
            Measures the quality of ranking.
          </li>
        </ul>
      </section>

      <section>
        <h2 id="model-comparison">Model Comparison and Selection</h2>
        <p>
          When comparing different models, it's important to use statistical
          tests to ensure the differences are significant:
        </p>
        <CodeBlock
          language="python"
          code={`
from scipy import stats

t_statistic, p_value = stats.ttest_rel(scores_model1, scores_model2)
print(f"T-statistic: {t_statistic}, p-value: {p_value}")
          `}
        />

        <h3>Ensemble Methods and Model Stacking</h3>
        <p>Combining multiple models can often lead to better performance:</p>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

# Base models
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Make predictions
rf_preds = cross_val_predict(rf, X, y, cv=5, method='predict_proba')
gb_preds = cross_val_predict(gb, X, y, cv=5, method='predict_proba')

# Stack predictions
stacked_preds = np.column_stack((rf_preds, gb_preds))

# Meta-model
meta_model = LogisticRegression()
meta_preds = cross_val_predict(meta_model, stacked_preds, y, cv=5)

print(f"Stacked Model Accuracy: {accuracy_score(y, meta_preds)}")
          `}
        />
      </section>

      <section>
        <h2 id="imbalanced-datasets">Dealing with Imbalanced Datasets</h2>
        <p>
          Imbalanced datasets can lead to misleading evaluation metrics. Here
          are some techniques to handle them:
        </p>
        <ul>
          <li>
            <strong>Oversampling:</strong> Increase the number of minority class
            samples.
          </li>
          <li>
            <strong>Undersampling:</strong> Decrease the number of majority
            class samples.
          </li>
          <li>
            <strong>SMOTE (Synthetic Minority Over-sampling Technique):</strong>{" "}
            Create synthetic samples of the minority class.
          </li>
        </ul>
        <CodeBlock
          language="python"
          code={`
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Create a pipeline with SMOTE and Random Under Sampling
imba_pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.5)),
    ('under', RandomUnderSampler(sampling_strategy=0.5))
])

# Fit and transform the data
X_resampled, y_resampled = imba_pipeline.fit_resample(X, y)
          `}
        />
      </section>

      <section>
        <h2 id="model-interpretability">
          Model Interpretability and Explainability
        </h2>
        <p>
          Understanding why a model makes certain predictions is crucial for
          many applications:
        </p>

        <h3>Feature Importance</h3>
        <p>Many models provide built-in feature importance scores:</p>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X, y)
importances = rf.feature_importances_
          `}
        />

        <h3>SHAP (SHapley Additive exPlanations) Values</h3>
        <p>
          SHAP values provide a unified measure of feature importance that works
          across many models:
        </p>
        <CodeBlock
          language="python"
          code={`
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
          `}
        />

        <h3>LIME (Local Interpretable Model-agnostic Explanations)</h3>
        <p>
          LIME explains individual predictions by approximating the model
          locally with an interpretable model:
        </p>
        <CodeBlock
          language="python"
          code={`
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names, mode='classification')
exp = explainer.explain_instance(X_test[0], model.predict_proba)
exp.show_in_notebook()
          `}
        />
      </section>
    </Container>
  );
};

export default ModelEvaluationValidation;
