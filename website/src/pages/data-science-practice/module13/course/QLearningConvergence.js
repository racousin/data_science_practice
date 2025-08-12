import React from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { Title, Text, Stack, Container, Accordion } from '@mantine/core';

const QLearningConvergence = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <Title order={1} id="q-learning-convergence" className="text-3xl font-bold mb-6">
        Q-Learning Convergence: Complete Proof
      </Title>

      {/* Preliminaries */}
      <section className="space-y-4">
        <Title order={2} className="text-2xl font-semibold">1. Preliminaries</Title>
        <Text>
          We denote a Markov decision process as a tuple (X, A, P, r), where:
        </Text>
        <ul className="list-disc pl-6 space-y-2 mt-2">
          <li>X is the (finite) state-space</li>
          <li>A is the (finite) action-space</li>
          <li>P represents the transition probabilities</li>
          <li>r represents the reward function</li>
        </ul>
        
        <div className="bg-blue-50 p-6 rounded-lg mt-4">
          <Text className="mb-4">
            The reward function r is defined over triplets (x, a, y):
          </Text>
          <BlockMath>{`r: X \\times A \\times X \\rightarrow \\mathbb{R}`}</BlockMath>
          <Text className="mt-4">
            assigning a reward r(x, a, y) everytime a transition from x to y occurs due to action a. 
            We admit r to be a bounded, deterministic function.
          </Text>
        </div>
      </section>

      {/* Value Function Definition */}
      <section className="space-y-4">
        <Title order={2} className="text-2xl font-semibold">2. Value Function</Title>
        <Text>
          The value of a state x is defined, for a sequence of controls {'{At}'}, as:
        </Text>
        <div className="bg-gray-50 p-6 rounded-lg">
          <BlockMath>{`J(x, \\{A_t\\}) = \\mathbb{E}\\left[\\sum_{t=0}^\\infty \\gamma^t R(X_t, A_t) | X_0 = x\\right]`}</BlockMath>
          
          <Text className="mt-4">The optimal value function is defined as:</Text>
          <BlockMath>{`V^*(x) = \\max_{A_t} J(x, \\{A_t\\})`}</BlockMath>
          
          <Text className="mt-4">and verifies:</Text>
          <BlockMath>{`V^*(x) = \\max_{a\\in A} \\sum_{y\\in X} P_a(x,y)[r(x,a,y) + \\gamma V^*(y)]`}</BlockMath>
        </div>
      </section>

      {/* Q-Function Definition */}
      <section className="space-y-4">
        <Title order={2} className="text-2xl font-semibold">3. Optimal Q-Function</Title>
        <Text>
          From here we define the optimal Q-function, Q* as:
        </Text>
        <div className="bg-gray-50 p-6 rounded-lg">
          <BlockMath>{`Q^*(x,a) = \\sum_{y\\in X} P_a(x,y)[r(x,a,y) + \\gamma V^*(y)]`}</BlockMath>
          
          <Text className="mt-4">The optimal Q-function is a fixed point of a contraction operator H:</Text>
          <BlockMath>{`(Hq)(x,a) = \\sum_{y\\in X} P_a(x,y)[r(x,a,y) + \\gamma \\max_{b\\in A} q(y,b)]`}</BlockMath>
        </div>
      </section>

      {/* Contraction Property */}
      <section className="space-y-4">
        <Title order={2} className="text-2xl font-semibold">4. Contraction Property</Title>
        <Text>
          This operator is a contraction in the sup-norm:
        </Text>
        <div className="bg-green-50 p-6 rounded-lg">
          <BlockMath>{`\\|Hq_1 - Hq_2\\|_\\infty \\leq \\gamma \\|q_1 - q_2\\|_\\infty`}</BlockMath>
          
          <Text className="mt-4">Proof:</Text>
          <BlockMath>{`\\begin{align*}
\\|Hq_1 - Hq_2\\|_\\infty &= \\max_{x,a} \\left|\\sum_{y\\in X} P_a(x,y)[r(x,a,y) + \\gamma \\max_{b\\in A} q_1(y,b) - r(x,a,y) - \\gamma \\max_{b\\in A} q_2(y,b)]\\right| \\\\
&= \\max_{x,a} \\gamma \\left|\\sum_{y\\in X} P_a(x,y)[\\max_{b\\in A} q_1(y,b) - \\max_{b\\in A} q_2(y,b)]\\right| \\\\
&\\leq \\max_{x,a} \\gamma \\sum_{y\\in X} P_a(x,y) |\\max_{b\\in A} q_1(y,b) - \\max_{b\\in A} q_2(y,b)| \\\\
&\\leq \\max_{x,a} \\gamma \\sum_{y\\in X} P_a(x,y) \\max_{z,b} |q_1(z,b) - q_2(z,b)| \\\\
&= \\gamma \\|q_1 - q_2\\|_\\infty
\\end{align*}`}</BlockMath>
        </div>
      </section>

      {/* Q-Learning Algorithm */}
      <section className="space-y-4">
        <Title order={2} className="text-2xl font-semibold">5. Q-Learning Algorithm</Title>
        <Text>
        Let <InlineMath>{`\\pi`}</InlineMath> be some random policy such that <InlineMath>{`\\mathbb{P}[A_t = a | X_t = x] > 0`}</InlineMath> for all state-action pairs <InlineMath>{`(x, a)`}</InlineMath>. 
        Given any initial estimate <InlineMath>{`Q_0`}</InlineMath>, Q-learning uses the following update rule:
      </Text>
        <div className="bg-yellow-50 p-6 rounded-lg">
          <BlockMath>{`Q_{t+1}(x_t,a_t) = Q_t(x_t,a_t) + \\alpha_t(x_t,a_t)[r_t + \\gamma \\max_{b\\in A} Q_t(x_{t+1},b) - Q_t(x_t,a_t)]`}</BlockMath>
          <Text className="mt-4">
        where the step-sizes <InlineMath>{`\\alpha_t(x,a)`}</InlineMath> verify <InlineMath>{`0 \\leq \\alpha_t(x,a) \\leq 1`}</InlineMath>
      </Text>
        </div>
      </section>

      {/* Convergence Theorem */}
      <section className="space-y-4">
        <Title order={2} className="text-2xl font-semibold">6. Convergence Theorem</Title>
        <div className="bg-purple-50 p-6 rounded-lg">
          <Text className="font-semibold">Theorem 1:</Text>
          <Text className="mt-2">
        Given a finite MDP <InlineMath>{`(\\mathcal{X}, \\mathcal{A}, P, r)`}</InlineMath>, the Q-learning algorithm converges with probability 1 to the optimal Q-function as long as:
      </Text>
          <BlockMath>{`\\sum_t \\alpha_t(x,a) = \\infty`}</BlockMath>
          <BlockMath>{`\\sum_t \\alpha_t^2(x,a) < \\infty`}</BlockMath>
          <Text className="mt-4">
        for all <InlineMath>{`(x,a) \\in \\mathcal{X} \\times \\mathcal{A}`}</InlineMath>.
      </Text>
        </div>
      </section>

      {/* Auxiliary Result */}
      <section className="space-y-4">
        <Title order={2} className="text-2xl font-semibold">7. Auxiliary Result</Title>
        <div className="bg-gray-50 p-6 rounded-lg">
          <Text className="font-semibold">Theorem 2:</Text>
          <Text className="mt-2">
          <Text>The random process <InlineMath>{`\\{\\Delta_t\\}`}</InlineMath> taking values in <InlineMath>{`\\mathbb{R}^n`}</InlineMath> and defined as:</Text>
          </Text>
          <BlockMath>{`\\Delta_{t+1}(x) = (1-\\alpha_t(x))\\Delta_t(x) + \\alpha_t(x)F_t(x)`}</BlockMath>
          <Text className="mt-4">converges to zero with probability 1 under the following assumptions:</Text>
          <ul className="list-disc pl-6 space-y-4">
      <li>
        <InlineMath>
          {`0 \\leq \\alpha_t \\leq 1, \\sum_t \\alpha_t(x) = \\infty \\text{ and } \\sum_t \\alpha_t^2(x) < \\infty`}
        </InlineMath>
      </li>
      <li>
        <InlineMath>
          {`\\|\\mathbb{E}[F_t(x) | \\mathcal{F}_t]\\|_W \\leq \\gamma \\|\\Delta_t\\|_W, \\text{ with } \\gamma < 1`}
        </InlineMath>
      </li>
      <li>
        <InlineMath>
          {`\\text{var}[F_t(x) | \\mathcal{F}_t] \\leq C(1 + \\|\\Delta_t\\|^2_W), \\text{ for } C > 0`}
        </InlineMath>
      </li>
    </ul>
        </div>
      </section>

      {/* Final Proof */}
      <section className="space-y-4">
        <Title order={2} className="text-2xl font-semibold">8. Final Proof</Title>
        <div className="bg-blue-50 p-6 rounded-lg">
          <Text>We start by rewriting the Q-learning update as:</Text>
          <BlockMath>{`\\Delta_t(x,a) = Q_t(x,a) - Q^*(x,a)`}</BlockMath>
          <Text className="mt-4">If we write:</Text>
          <BlockMath>{`F_t(x,a) = r(x,a,X(x,a)) + \\gamma \\max_{b\\in A} Q_t(y,b) - Q^*(x,a)`}</BlockMath>
          <Text className="mt-4">where X(x,a) is a random sample state, we have:</Text>
          <BlockMath>{`\\mathbb{E}[F_t(x,a)|\\mathcal{F}_t] = (HQ_t)(x,a) - Q^*(x,a)`}</BlockMath>
          <Text className="mt-4">Using the contraction property:</Text>
          <BlockMath>{`\\|\\mathbb{E}[F_t(x,a)|\\mathcal{F}_t]\\|_\\infty \\leq \\gamma \\|Q_t - Q^*\\|_\\infty = \\gamma \\|\\Delta_t\\|_\\infty`}</BlockMath>
          <Text className="mt-4">Therefore, by Theorem 2, Δt converges to zero with probability 1, i.e., Qt converges to Q* with probability 1.</Text>
        </div>
      </section>
    </div>
  );
};

export default QLearningConvergence;