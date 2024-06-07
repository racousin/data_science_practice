import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CourseGenerativeModels = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Generative Models",
      component: lazy(() => import("pages/module14/course/Introduction")),
    },
    {
      to: "/variational-autoencoders",
      label: "Variational Autoencoders (VAEs)",
      component: lazy(() =>
        import("pages/module14/course/VariationalAutoencoders")
      ),
    },
    {
      to: "/generative-adversarial-networks",
      label: "Generative Adversarial Networks (GANs)",
      component: lazy(() =>
        import("pages/module14/course/GenerativeAdversarialNetworks")
      ),
    },
    {
      to: "/advanced-gan-applications",
      label: "Advanced GAN Applications",
      component: lazy(() =>
        import("pages/module14/course/AdvancedGANApplications")
      ),
    },
    {
      to: "/other-generative-models",
      label: "Other Generative Models",
      component: lazy(() =>
        import("pages/module14/course/OtherGenerativeModels")
      ),
    },
    {
      to: "/evaluation-and-enhancement",
      label: "Evaluation and Enhancement of Generative Models",
      component: lazy(() =>
        import("pages/module14/course/EvaluationAndEnhancement")
      ),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={14}
          isCourse={true}
          title="Module 14: Generative Models"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about generative models and their
          applications in AI.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module14/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseGenerativeModels;
