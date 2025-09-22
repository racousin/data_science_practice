import React from 'react';
import { Container, Title, Text, Stack, Paper, List, Grid } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const ModelsObjective = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">

        {/* Slide 1: ML Objective */}
        <div data-slide>
          <Title order={1} mb="xl">
            Machine Learning Objective
          </Title>

          <Text size="lg" mb="md">
            Machine learning seeks to find a function <InlineMath>{`f`}</InlineMath> that maps inputs <InlineMath>{`X`}</InlineMath> to outputs <InlineMath>{`Y`}</InlineMath>.
          </Text>

          <Text size="lg" mb="md">
            Given training data <InlineMath>{`\\{(x_i, y_i)\\}_{i=1}^n`}</InlineMath>, we want to learn:
          </Text>

          <BlockMath>{`f: \\mathbb{R}^d \\to \\mathbb{R}^k`}</BlockMath>

          <Text size="lg" mb="md">
            Such that <InlineMath>{`f(x) \\approx y`}</InlineMath> for new, unseen data.
          </Text>


        </div>

        {/* Slide 2: Models */}
        <div data-slide>
          <Title order={1} mb="xl">
            Models
          </Title>

          <Text size="lg" mb="md">
            A model defines the space of possible functions we can learn.
          </Text>

          <Paper p="md" mb="md">
            <Title order={3} mb="sm">Parametric Models</Title>
            <Text mb="xs">Functions with fixed number of parameters:</Text>
            <BlockMath>{`f_\\theta(x) = \\text{function}(x, \\theta)`}</BlockMath>
            <Text size="sm">where <InlineMath>{`\\theta \\in \\mathbb{R}^p`}</InlineMath> are learnable parameters</Text>
            <Text size="lg">
              Model is training using gradient descent on loss function <InlineMath>{`\\ell(y, f(x))`}</InlineMath>.
            </Text>

          </Paper>

          <Paper p="md">
            <Title order={3} mb="sm">Non-Parametric Models</Title>
            <Text mb="xs">Functions where complexity grows with data:</Text>
            <BlockMath>{`f \\in \\mathcal{F}`}</BlockMath>
            <Text size="sm">where <InlineMath>{`\\mathcal{F}`}</InlineMath> is a flexible function space</Text>
            <Text size="lg">
              Model is trained by selecting the best function in <InlineMath>{`\\mathcal{F}`}</InlineMath> to minimize loss <InlineMath>{`\\ell(y, f(x))`}</InlineMath>.
            </Text>
          </Paper>
        </div>

        {/* Slide 3: Parametric vs Non-Parametric */}
        <div data-slide>
          <Title order={1} mb="xl">
            Parametric vs Non-Parametric Models
          </Title>

          <Paper p="md" mb="md">
            <Title order={3} mb="sm">Parametric Models</Title>
            <List spacing="sm">
              <List.Item>Fixed number of parameters <InlineMath>{`p`}</InlineMath></List.Item>
              <List.Item>Learning: optimize <InlineMath>{`\\theta^* = \\arg\\min_\\theta \\ell(\\theta)`}</InlineMath></List.Item>
              <List.Item>Examples: Linear regression, Neural networks</List.Item>
              <List.Item>Fast prediction, limited flexibility</List.Item>
            </List>
          </Paper>

          <Paper p="md">
            <Title order={3} mb="sm">Non-Parametric Models</Title>
            <List spacing="sm">
              <List.Item>Parameters grow with data size <InlineMath>{`n`}</InlineMath></List.Item>
              <List.Item>Learning: optimize <InlineMath>{`f^* = \\arg\\min_{f \\in \\mathcal{F}} \\ell(f)`}</InlineMath></List.Item>
              <List.Item>Examples: K-NN, Decision trees, Kernel methods</List.Item>
              <List.Item>High flexibility, potentially slower prediction</List.Item>
            </List>
          </Paper>
        </div>
 <div data-slide>
        <Title order={1} mb="lg">Model Training and Prediction</Title>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
            </marker>
          </defs>
          <rect x="10" y="10" width="180" height="80" rx="5" fill="#e6f3ff" stroke="#333" stroke-width="2"/>
          <text x="100" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Historical Data</text>
          <text x="100" y="75" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">(Features X, Labels y)</text>
          <line x1="190" y1="50" x2="240" y2="50" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
          <rect x="240" y="10" width="140" height="80" rx="5" fill="#fff2e6" stroke="#333" stroke-width="2"/>
          <text x="310" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Data Processing</text>
          <line x1="380" y1="50" x2="430" y2="50" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
          <rect x="430" y="10" width="140" height="80" rx="5" fill="#e6ffe6" stroke="#333" stroke-width="2"/>
          <text x="500" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Model Training</text>
          <line x1="570" y1="50" x2="620" y2="50" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
          <rect x="620" y="10" width="140" height="80" rx="5" fill="#ffe6e6" stroke="#333" stroke-width="2"/>
          <text x="690" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Trained Model</text>
          <rect x="10" y="250" width="180" height="80" rx="5" fill="#e6f3ff" stroke="#333" stroke-width="2"/>
          <text x="100" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">New Data</text>
          <text x="100" y="315" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">(Features X)</text>
          <line x1="190" y1="290" x2="240" y2="290" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
          <rect x="240" y="250" width="140" height="80" rx="5" fill="#fff2e6" stroke="#333" stroke-width="2"/>
          <text x="310" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Data Processing</text>
          <line x1="380" y1="290" x2="430" y2="290" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
          <rect x="430" y="250" width="140" height="80" rx="5" fill="#ffe6e6" stroke="#333" stroke-width="2"/>
          <text x="500" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Trained Model</text>
          <line x1="570" y1="290" x2="620" y2="290" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
          <rect x="620" y="250" width="140" height="80" rx="5" fill="#e6ffe6" stroke="#333" stroke-width="2"/>
          <text x="690" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Predictions</text>
          <path d="M690 90 L690 170 Q690 190 670 190 L480 190 Q460 190 460 210 L460 250" fill="none" stroke="#333" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrowhead)"/>
          <text x="400" y="140" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold">Training Process</text>
          <text x="400" y="380" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold">Serving/Predicting Process</text>
        </svg>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col span={{ md: 6 }}>
            <Title order={2} id="model-fitting" mb="md">Training</Title>
            <Text mb="md">
              In supervised learning, we typically have:
            </Text>
            <List spacing="xs" mb="md">
              <List.Item>
                <InlineMath math="X \in \mathbb{R}^{n \times p}" />: Input features (n
                samples, p features)
              </List.Item>
              <List.Item>
                <InlineMath math="y \in \mathbb{R}^n" />: Target variable
              </List.Item>
              <List.Item>
                <InlineMath math="f: \mathbb{R}^p \rightarrow \mathbb{R}" />: The true
                function we're trying to approximate
              </List.Item>
              <List.Item>
                <InlineMath math="\hat{f}: \mathbb{R}^p \rightarrow \mathbb{R}" />:
                Our model's approximation of f
              </List.Item>
            </List>
            <Text mb="md">
              The goal is to find <InlineMath math="\hat{f}" /> that minimizes some
              loss function <InlineMath math="L(y, \hat{f}(X))" />.
            </Text>
            <Text mb="md">Here's an example of model training using scikit-learn:</Text>
            <CodeBlock
              language="python"
              code={`# 1. Data Collection
X_train = np.random.rand(100, 5)
y_train = np.sum(X_train, axis=1) + np.random.randn(100) * 0.1
# 2. Data Processing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 4. Model Saving
joblib.dump(model, 'linear_model.joblib')
joblib.dump(scaler, 'scaler.joblib')`}
            />
          </Grid.Col>
          <Grid.Col span={{ md: 6 }}>
            <Title order={2} id="prediction" mb="md">Prediction</Title>
            <Text mb="md">
              Once the model is trained, we can use it to make predictions on new data. This process involves:
            </Text>
            <List type="ordered" spacing="xs" mb="md">
              <List.Item>Loading the saved model</List.Item>
              <List.Item>Processing new data in the same way as the training data</List.Item>
              <List.Item>Using the model to generate predictions</List.Item>
            </List>
            <Text mb="md">Here's an example of using the trained model for prediction:</Text>
            <CodeBlock
              language="python"
              code={`# Model Serving
# 1. Load the model and scaler
loaded_model = joblib.load('linear_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

# 2. Process new data
X_new = np.random.rand(10, 5)
X_new_scaled = loaded_scaler.transform(X_new)

# 3. Make predictions
predictions = loaded_model.predict(X_new_scaled)
print("Predictions:", predictions)`}
            />
          </Grid.Col>
        </Grid>
      </div>


      <div data-slide>
        <Title order={2} id="considerations" mb="md">Key Considerations</Title>
        <List spacing="sm">
          <List.Item>Ensure consistency in data processing between training and prediction phases</List.Item>
          <List.Item>Regularly retrain models with new data to maintain performance</List.Item>
          <List.Item>Monitor model performance in production to detect drift or degradation</List.Item>
          <List.Item>Consider model versioning for tracking changes and facilitating rollbacks if needed</List.Item>
        </List>
      </div>
      </Stack>
    </Container>
  );
};

export default ModelsObjective;