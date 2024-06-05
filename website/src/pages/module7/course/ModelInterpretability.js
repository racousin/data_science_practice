import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ModelInterpretability = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Model Interpretability</h1>
      <p>
        In this section, you will learn about methods for interpreting complex
        models trained on tabular data.
      </p>
      <Row>
        <Col>
          <h2>Techniques for Model Interpretability (SHAP, LIME)</h2>
          <p>
            SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable
            Model-agnostic Explanations) are two popular techniques for
            interpreting complex models. SHAP measures the contribution of each
            feature to the prediction of a single instance, while LIME generates
            a local explanation for a single instance by fitting a simple model
            to the neighborhood of that instance.
          </p>
          <CodeBlock
            code={`# Example of SHAP
import shap

explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.plots.waterfall(shap_values[0])`}
          />
          <h2>Importance of Model Transparency in Business Applications</h2>
          <p>
            Model transparency is important in business applications, as it
            allows stakeholders to understand how the model is making
            predictions and to trust the model's recommendations. Interpretable
            models can help to build trust with customers, regulators, and other
            stakeholders.
          </p>
          <h2>Visualizing Feature Importances and Decision Paths</h2>
          <p>
            Visualizing feature importances and decision paths can help to gain
            insights into how the model is making predictions. This can be
            particularly useful for complex models, such as deep neural networks
            and ensemble methods.
          </p>
          <CodeBlock
            code={`# Example of visualizing feature importances
importances = model.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default ModelInterpretability;
