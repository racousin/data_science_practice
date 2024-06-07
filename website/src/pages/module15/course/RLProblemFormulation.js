import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const RLProblemFormulation = () => {
  return (
    <Container fluid>
      <h1 className="my-4">RL Problem Formulation</h1>
      <p>
        In this section, you will learn how to formally define and model
        reinforcement learning problems.
      </p>
      <Row>
        <Col>
          <h2>Markov Decision Processes (MDPs)</h2>
          <p>
            Here, we'll discuss what Markov Decision Processes are and how they
            are used in reinforcement learning.
          </p>

          <h2>Components of MDPs</h2>
          <p>
            We'll cover the key components of Markov Decision Processes,
            including transition probabilities and reward functions.
          </p>
          <CodeBlock
            code={`
// Example code for a simple MDP
class MDP {
  constructor(states, actions, transitionProbabilities, rewardFunction) {
    this.states = states;
    this.actions = actions;
    this.transitionProbabilities = transitionProbabilities;
    this.rewardFunction = rewardFunction;
  }

  // Get the possible actions for a given state
  possibleActions(state) {
    // This assumes that the same actions are possible in all states,
    // you'd replace this with your own logic if that's not the case
    return this.actions;
  }

  // Step the MDP: take an action and transition to a new state
  step(state, action) {
    const nextState = this.sampleNextState(state, action);
    const reward = this.rewardFunction(state, action, nextState);
    return { nextState, reward, reward };
  }

  // Sample a next state according to the transition probabilities
  sampleNextState(state, action) {
    const probabilities = this.transitionProbabilities[state][action];
    const cumulativeProbabilities = probabilities.map((p, i) => (i > 0 ? probabilities[i - 1] + p : p));
    const randomValue = Math.random();
    const nextStateIndex = cumulativeProbabilities.findIndex(p => p >= randomValue);
    return this.states[nextStateIndex];
  }
}
            `}
          />

          <h2>The Balance between Exploration and Exploitation</h2>
          <p>
            We'll explore the trade-off between exploring new actions and
            exploiting the ones that have worked well in the past.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default RLProblemFormulation;
