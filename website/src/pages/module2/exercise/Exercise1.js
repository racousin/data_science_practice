import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Exercise1 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 1: Creating a Python Function</h1>
      <p>
        In this exercise, you will create a Python function that takes two
        arguments and returns their product if both are numbers; otherwise, it
        returns the string "error".
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>Create a new directory called `mysupertools`:</li>
            <CodeBlock code={`mkdir mysupertools`} />
            <li>Navigate to the `mysupertools` directory:</li>
            <CodeBlock code={`cd mysupertools`} />
            <li>Create a new directory called `tool`:</li>
            <CodeBlock code={`mkdir tool`} />
            <li>
              Create a new file called `multiplication_a_b.py` inside the `tool`
              directory:
            </li>
            <CodeBlock code={`touch tool/multiplication_a_b.py`} />
            <li>
              Open `multiplication_a_b.py` in a text editor and add the
              following code:
            </li>
            <CodeBlock
              code={`def multiply(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    else:
        return "error"

if __name__ == "__main__":
    print(multiply(2, 3))  # Output: 6
    print(multiply(2, "3"))  # Output: error`}
            />
          </ol>
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise1;
