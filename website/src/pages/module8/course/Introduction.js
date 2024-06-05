import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Deep Learning</h1>
      <p>
        In this section, you will learn about the fundamentals of deep learning
        and its distinction from traditional machine learning.
      </p>
      <Row>
        <Col>
          <h2>History and Evolution of Neural Networks</h2>
          <p>
            Neural networks have been around for several decades, but they have
            gained popularity in recent years due to the availability of large
            amounts of data and powerful computing resources. The first neural
            networks were inspired by the structure and function of the human
            brain, and they were used for tasks such as pattern recognition and
            image classification.
          </p>
          <h2>Key Concepts: Neurons, Layers, Activation Functions</h2>
          <p>
            Neural networks are composed of interconnected nodes called neurons.
            Neurons receive input signals from other neurons, process them using
            a mathematical function called an activation function, and produce
            an output signal that is sent to other neurons. Neural networks are
            organized into layers, with each layer containing multiple neurons
            that receive input from the neurons in the previous layer and send
            output to the neurons in the next layer.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
