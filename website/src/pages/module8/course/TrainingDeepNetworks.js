import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const TrainingDeepNetworks = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Training Deep Networks</h1>
      <p>
        In this section, you will learn about techniques for training deep
        neural networks effectively.
      </p>
      <Row>
        <Col>
          <h2>Optimizing Learning Rate and Other Hyperparameters</h2>
          <p>
            The learning rate is a hyperparameter that controls how quickly the
            model learns. Other hyperparameters that can be optimized include
            the batch size, the number of epochs, and the regularization
            strength.
          </p>
          <h2>Techniques to Combat Overfitting: Regularization, Dropout</h2>
          <p>
            Overfitting occurs when a model learns the training data too well
            and performs poorly on unseen data. Regularization techniques such
            as L1 and L2 regularization can be used to prevent overfitting.
            Dropout is a technique that randomly drops out a fraction of the
            neurons in a layer during training, which can help to prevent
            overfitting.
          </p>
          <CodeBlock
            code={`# Example of using L2 regularization in PyTorch
import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleFCN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)`}
          />
          <h2>Using Callbacks and Monitoring Training with TensorBoard</h2>
          <p>
            Callbacks are functions that are called at various points during
            training. They can be used to save checkpoints, log metrics, and
            early stop training if the model stops improving. TensorBoard is a
            visualization tool that can be used to monitor training metrics and
            visualize the model's architecture.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default TrainingDeepNetworks;
