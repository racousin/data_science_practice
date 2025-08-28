import React from 'react';
import { Container, Title, Text, Stack, List, Table, Group } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const CustomObjectivesGuide = () => {
  return (
    <Container fluid>
      <Stack spacing="xl">
        <Title order={1} id="custom-objectives">Custom Objectives Guide</Title>

        <Title order={3} id="understanding-objective">Understanding the Objective</Title>
        <Text mb="md">Before modifying model behavior, it's crucial to clearly define what constitutes a 'good' prediction in your specific context.</Text>
          <Stack spacing="lg">
            <Table withBorder withColumnBorders>
              <thead>
                <tr>
                  <th>Consideration</th>
                  <th>Questions to Ask</th>
                  <th>Impact on Model</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Error Cost Structure</td>
                  <td>
                    - Are errors equally costly?<br/>
                    - Are there critical thresholds?<br/>
                    - Do classes have different importance?
                  </td>
                  <td>
                    - Custom loss functions<br/>
                    - Sample weights<br/>
                    - Class weights
                  </td>
                </tr>
                <tr>
                  <td>Target Distribution</td>
                  <td>
                    - Is the target balanced?<br/>
                    - Are rare cases important?<br/>
                    - Will distribution shift?
                  </td>
                  <td>
                    - Sampling techniques<br/>
                    - Robust loss functions<br/>
                    - Distribution-aware validation
                  </td>
                </tr>
              </tbody>
            </Table>
          </Stack>

        <Title order={3} id="customization-techniques">Customization Techniques</Title>
          <Stack spacing="lg">
            <SubSection
              title="Sample Weights"
              description="Control the importance of individual samples during training."
            >
              <CodeBlock
                language="python"
                code={`
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create weights based on importance
sample_weights = np.where(
    y > threshold,
    2.0,  # Higher weight for important samples
    1.0   # Normal weight for regular samples
)

# Train with weights
model = RandomForestClassifier()
model.fit(X, y, sample_weight=sample_weights)
`}
              />
            </SubSection>

            <SubSection
              title="Custom Objectives"
              description="Define exact optimization criteria for your model."
            >
              <CodeBlock
                language="python"
                code={`
import xgboost as xgb

def custom_asymmetric_objective(y_true, y_pred):
    """Asymmetric objective penalizing under-predictions more"""
    diff = y_true - y_pred
    grad = np.where(diff > 0, 
                    2.0 * diff,    # Higher gradient for under-predictions
                    0.5 * diff)    # Lower gradient for over-predictions
    hess = np.where(diff > 0, 2.0, 0.5)
    return grad, hess

# Train with custom objective
params = {
    'objective': custom_asymmetric_objective,
    'eval_metric': 'mae'
}
model = xgb.train(params, dtrain)
`}
              />
            </SubSection>
          </Stack>
        

        <Title order={3} id="model-combinations">Model Combinations</Title>
          <Stack spacing="lg">
            <Table withBorder withColumnBorders>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Description</th>
                  <th>Use Cases</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Model Stacking</td>
                  <td>Combine predictions from multiple base models using a meta-model</td>
                  <td>Complex objectives with multiple criteria</td>
                </tr>
                <tr>
                  <td>Expert Models</td>
                  <td>Separate models for different data segments</td>
                  <td>Segment-specific requirements</td>
                </tr>
                <tr>
                  <td>Hierarchical</td>
                  <td>Models arranged in a decision tree structure</td>
                  <td>Multiple levels of objectives</td>
                </tr>
              </tbody>
            </Table>

            <List spacing="md">
              <List.Item>
                <Text weight={500}>Best Practice: Model Diversity</Text>
                <Text size="sm">Choose models with different strengths to capture various aspects of the objective</Text>
              </List.Item>
              <List.Item>
                <Text weight={500}>Best Practice: Complexity Balance</Text>
                <Text size="sm">Ensure added complexity is justified by performance improvements</Text>
              </List.Item>
            </List>
          </Stack>
        
      </Stack>
    </Container>
  );
};



const SubSection = ({ title, description, children }) => (
  <Stack spacing="sm">
    <Title order={3}>{title}</Title>
    <Text>{description}</Text>
    {children}
  </Stack>
);

export default CustomObjectivesGuide;