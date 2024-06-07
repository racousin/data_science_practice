import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CourseReinforcementLearning = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Reinforcement Learning",
      component: lazy(() => import("pages/module15/course/Introduction")),
    },
    {
      to: "/rl-problem-formulation",
      label: "RL Problem Formulation",
      component: lazy(() =>
        import("pages/module15/course/RLProblemFormulation")
      ),
    },
    {
      to: "/value-based-methods",
      label: "Value-Based Methods",
      component: lazy(() => import("pages/module15/course/ValueBasedMethods")),
    },
    {
      to: "/policy-based-methods",
      label: "Policy-Based Methods",
      component: lazy(() => import("pages/module15/course/PolicyBasedMethods")),
    },
    {
      to: "/model-free-and-model-based-rl",
      label: "Model-Free and Model-Based RL",
      component: lazy(() =>
        import("pages/module15/course/ModelFreeAndModelBasedRL")
      ),
    },
    {
      to: "/deep-reinforcement-learning",
      label: "Deep Reinforcement Learning",
      component: lazy(() =>
        import("pages/module15/course/DeepReinforcementLearning")
      ),
    },
    {
      to: "/real-world-applications-and-challenges",
      label: "Real-World Applications and Challenges",
      component: lazy(() =>
        import("pages/module15/course/RealWorldApplicationsAndChallenges")
      ),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={15}
          isCourse={true}
          title="Module 15: Reinforcement Learning"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about reinforcement learning and its
          applications in AI.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module15/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseReinforcementLearning;
