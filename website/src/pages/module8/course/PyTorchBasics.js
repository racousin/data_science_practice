import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const PyTorchBasics = () => {
  return (
    <Container fluid>
      <h1 className="my-4">PyTorch Basics</h1>
      <p>
        In this section, you will learn about PyTorch as a tool for deep
        learning.
      </p>
      <Row>
        <Col>
          <h2>Introduction to PyTorch and its Ecosystem</h2>
          <p>
            PyTorch is an open-source machine learning library developed by
            Facebook. It is known for its flexibility, ease of use, and support
            for dynamic computation graphs. PyTorch is built on top of the Torch
            library and is used by many researchers and practitioners in the
            field of deep learning.
          </p>
          <h2>Tensors and Operations in PyTorch</h2>
          <p>
            Tensors are multi-dimensional arrays that are used to represent data
            in PyTorch. Tensors support a variety of mathematical operations,
            including element-wise operations, matrix multiplication, and linear
            algebra operations.
          </p>
          <CodeBlock
            code={`# Example of creating a tensor
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = x + y
print(z)`}
          />
          <h2>Autograd System for Automatic Differentiation</h2>
          <p>
            The autograd system in PyTorch is used for automatic
            differentiation. It allows PyTorch to compute the gradients of
            tensors with respect to other tensors, which is necessary for
            training neural networks using gradient descent.
          </p>
          <CodeBlock
            code={`# Example of using autograd
x = torch.tensor([1, 2, 3], requires_grad=True)
y = torch.tensor([4, 5, 6], requires_grad=True)
z = (x * y).sum()
z.backward()
print(x.grad)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default PyTorchBasics;
