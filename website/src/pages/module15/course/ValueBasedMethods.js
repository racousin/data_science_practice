import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ValueBasedMethods = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Value-Based Methods</h1>
      <p>
        In this section, you will explore value-based methods in reinforcement
        learning.
      </p>
      <Row>
        <Col>
          <h2>Dynamic Programming in RL</h2>
          <p>
            Here, we'll discuss what dynamic programming is and how it is used
            in reinforcement learning.
          </p>

          <h2>Algorithms: Value Iteration and Policy Iteration</h2>
          <p>
            We'll cover two key dynamic programming algorithms: value iteration
            and policy iteration.
          </p>
          <CodeBlock
            code={`
// Example code for value iteration
function valueIteration(MDP, gamma, theta) {
  const states = MDP.states;
  const V = new Array(states.length).fill(0);

  while (true) {
    const delta = new Array(states.length).fill(0);

    for (let i = 0; i < states.length; i++) {
      const state = states[i];
      const actions = MDP.possibleActions(state);
      let v = -Infinity;

      for (let j = 0; j < actions.length; j++) {
        const action = actions[j];
        const q = MDP.qValue(state, action, V, gamma);
        v = Math.max(v, q);
      }

      delta[i] = Math.abs(v - V[i]);
      V[i] = v;
    }

    if (delta.reduce((a, b) => a + b, 0) < theta) {
      break;
    }
  }

  return V;
}

// Example code for policy iteration
function policyIteration(MDP, gamma, theta) {
  const states = MDP.states;
  const policy = new Array(states.length);
  const V = new Array(states.length).fill(0);

  while (true) {
    // Policy evaluation
    let delta = 0;

    for (let i = 0; i < states.length; i++) {
      const state = states[i];
      const action = policy[i];
      const v = MDP.qValue(state, action, V, gamma);
      delta = Math.max(delta, Math.abs(v - V[i]));
      V[i] = v;
    }

    if (delta < theta) {
      break;
    }

    // Policy improvement
    for (let i = 0; i < states.length; i++) {
      const state = states[i];
      const actions = MDP.possibleActions(state);
      let bestAction = null;
      let bestValue = -Infinity;

      for (let j = 0; j < actions.length; j++) {
        const action = actions[j];
        const value = MDP.qValue(state, action, V, gamma);

        if (value > bestValue) {
          bestValue = value;
          bestAction = action;
        }
      }

      policy[i] = bestAction;
    }
  }

  return policy;
}
            `}
          />

          <h2>Q-learning and its Variations</h2>
          <p>
            We'll explore Q-learning, a model-free reinforcement learning
            algorithm, and some of its variations.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default ValueBasedMethods;
