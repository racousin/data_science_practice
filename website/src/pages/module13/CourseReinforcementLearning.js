import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseReinforcementLearning = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction",
      component: lazy(() => import("pages/module13/course/Introduction")),
      subLinks: [
        { id: "ml-paradigms", label: "Machine Learning Paradigms" },
        { id: "rl-framework", label: "RL Framework" },
        { id: "applications", label: "Applications" },
        { id: "limitations", label: "Limitations and Challenges" }
      ],
    },
    {
      to: "/mdp",
      label: "Markov Decision Processes",
      component: lazy(() => import("pages/module13/course/MDP")),
      subLinks: [
        { id: "state-space", label: "State Space" },
        { id: "action-space", label: "Action Space" },
        { id: "transition-model", label: "Transition Model" },
        { id: "reward-function", label: "Reward Function" },
        { id: "policy", label: "Policy" },
        { id: "value-function", label: "Value Function" },
        { id: "bellman-equations", label: "Bellman Equations" },
      ],
    },
    {
      to: "/dynamic-programming",
      label: "Dynamic Programming",
      component: lazy(() => import("pages/module13/course/DynamicProgramming")),
      subLinks: [
        { id: "bellman-equations", label: "Bellman Equations" },
        { id: "optimal-policy", label: "Optimal Policy" },
        { id: "mdp-solution", label: "MDP Solution" },
        { id: "policy-evaluation", label: "Policy Evaluation" },
        { id: "policy-improvement", label: "Policy Improvement" },
        { id: "implementation", label: "Implementation" }
      ],
    },
    {
      to: "/rl-paradigms",
      label: "RL Paradigms",
      component: lazy(() => import("pages/module13/course/RLParadigms")),
      subLinks: [
        { id: "mdp-connection", label: "From MDP to RL" },
        { id: "exploration-exploitation", label: "Exploration vs Exploitation" },
        { id: "bandit-vs-rl", label: "From Bandits to RL" },
        { id: "model-based-vs-free", label: "Model-Based vs Model-Free" },
        { id: "epsilon-greedy", label: "ε-Greedy Strategy" }
      ],
    },
    {
      to: "/model-free-methods",
      label: "Model-Free Methods",
      component: lazy(() => import("pages/module13/course/ModelFreeMethods")),
      subLinks: [
        { id: "monte-carlo-algorithm", label: "Monte Carlo Algorithm" },
        { id: "td-learning", label: "Temporal Difference Learning" },
        { id: "sarsa", label: "SARSA" },
        { id: "q-learning", label: "Q-Learning" },
      ],
    },
    {
      to: "/deep-model-free",
      label: "Deep Model Free",
      component: lazy(() => import("pages/module13/course/DeepModelFree")),
      subLinks: [
        { id: "deep-q-learning", label: "Deep Q-Learning" },
        { id: "policy-gradient", label: "Policy Gradient Methods" },
        { id: "actor-critic", label: "Actor-Critic Methods" },
      ],
    },
    // {
    //   to: "/model-based",
    //   label: "Model-Based Methods",
    //   component: lazy(() => import("pages/module13/course/ModelBasedMethods")),
    // },
    {
      to: "/rl-training-efficiency",
      label: "RL Training Efficiency",
      component: lazy(() => import("pages/module13/course/RLTrainingEfficiency")),
      subLinks: [
        { id: "gymnasium-basics", label: "Training with Gymnasium" },
        { id: "pettingzoo-basics", label: "Multi-Agent Training" },
        { id: "vectorized-training", label: "Vectorized Training" },
        { id: "monitoring", label: "Monitoring and Optimization" }
      ],
    }
  ];

  const location = useLocation();
  const module = 13;
  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 13: Reinforcement Learning"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <Col>
              <p>Last Updated: {"2025-01-23"}</p>
            </Col>
          </Row>
        </>
      )}
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseReinforcementLearning;
