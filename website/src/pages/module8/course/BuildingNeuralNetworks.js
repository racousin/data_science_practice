import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const BuildingNeuralNetworks = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Building Neural Networks</h1>
      <p>
        In this section, you will learn how to build basic neural network
        structures using PyTorch.
      </p>
      <Row>
        <Col>
          <h2>Designing Simple Fully Connected Networks (FCNs)</h2>
          <p>
            Fully connected networks (FCNs) are the simplest type of neural
            network. They consist of an input layer, one or more hidden layers,
            and an output layer. Each neuron in a fully connected layer is
            connected to every neuron in the previous layer.
          </p>
          <h2>Using PyTorch Modules like nn.Module, nn.Linear</h2>
          <p>
            PyTorch provides a variety of modules that can be used to build
            neural networks. The `nn.Module` class is the base class for all
            neural network modules in PyTorch. The `nn.Linear` module is used to
            create a fully connected layer.
          </p>
          <CodeBlock
            code={`# Example of building a simple FCN
import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x`}
          />
          <h2>Implementing Common Activation Functions</h2>
          <p>
            Activation functions are used to introduce non-linearity into neural
            networks. Common activation functions include the rectified linear
            unit (ReLU), the sigmoid function, and the hyperbolic tangent
            function.
          </p>
          <CodeBlock
            code={`# Example of implementing the ReLU activation function
def relu(x):
    return torch.max(x, torch.tensor(0.0))`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default BuildingNeuralNetworks;
