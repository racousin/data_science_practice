import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const RecurrentNeuralNetworksLSTMs = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Recurrent Neural Networks (RNNs) and LSTMs</h1>
      <p>
        In this section, you will learn about RNNs and LSTMs, which are suitable
        for processing sequential data.
      </p>
      <Row>
        <Col>
          <h2>Structure and Functioning of RNNs and LSTMs</h2>
          <p>
            RNNs are a type of neural network that are designed to process
            sequential data. They have a recurrent connection that allows
            information to flow from one time step to the next. LSTMs are a type
            of RNN that are designed to overcome the vanishing gradient problem,
            which can occur when training RNNs on long sequences of data.
          </p>
          <h2>
            Applications in Time-Series Analysis and Natural Language Processing
          </h2>
          <p>
            RNNs and LSTMs are widely used in time-series analysis and natural
            language processing. They can be used to predict future values in a
            time series, to generate text, and to classify text.
          </p>
          <CodeBlock
            code={`# Example of building an LSTM for text classification
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out`}
          />
          <h2>
            Handling Challenges like Long-Term Dependencies and Vanishing
            Gradients
          </h2>
          <p>
            RNNs and LSTMs can struggle to learn long-term dependencies in data.
            This can occur when the gap between the relevant information and the
            current time step is too large. LSTMs are designed to overcome this
            problem by using a mechanism called a "gate" that allows information
            to flow through the network unchanged.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default RecurrentNeuralNetworksLSTMs;
