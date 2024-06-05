import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ConvolutionalNeuralNetworks = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Convolutional Neural Networks (CNNs)</h1>
      <p>
        In this section, you will learn about CNN architectures and their
        applications in image processing.
      </p>
      <Row>
        <Col>
          <h2>Basics of Convolutional Layers and Pooling Layers</h2>
          <p>
            Convolutional layers are used to extract features from input data,
            such as images. Pooling layers are used to reduce the dimensionality
            of the data and make the model more robust to variations in the
            input.
          </p>
          <h2>Building a CNN for Image Classification</h2>
          <p>
            CNNs are particularly well-suited for image classification tasks.
            They consist of multiple convolutional layers, followed by one or
            more fully connected layers.
          </p>
          <CodeBlock
            code={`# Example of building a CNN for image classification
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x`}
          />
          <h2>Understanding and Visualizing What CNNs Learn</h2>
          <p>
            Visualizing what CNNs learn can help to gain insights into how they
            process input data. Techniques such as activation visualization and
            feature visualization can be used to visualize the activations of
            individual neurons and the features that the model has learned to
            extract.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default ConvolutionalNeuralNetworks;
