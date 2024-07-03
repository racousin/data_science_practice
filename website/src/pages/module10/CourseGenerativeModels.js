import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseGenerativeModels = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Generative Models",
      component: lazy(() => import("pages/module10/course/Introduction")),
    },
    {
      to: "/variational-autoencoders",
      label: "Variational Autoencoders (VAEs)",
      component: lazy(() =>
        import("pages/module10/course/VariationalAutoencoders")
      ),
    },
    {
      to: "/generative-adversarial-networks",
      label: "Generative Adversarial Networks (GANs)",
      component: lazy(() =>
        import("pages/module10/course/GenerativeAdversarialNetworks")
      ),
    },
    {
      to: "/advanced-gan-applications",
      label: "Advanced GAN Applications",
      component: lazy(() =>
        import("pages/module10/course/AdvancedGANApplications")
      ),
    },
    {
      to: "/other-generative-models",
      label: "Other Generative Models",
      component: lazy(() =>
        import("pages/module10/course/OtherGenerativeModels")
      ),
    },
    {
      to: "/evaluation-and-enhancement",
      label: "Evaluation and Enhancement of Generative Models",
      component: lazy(() =>
        import("pages/module10/course/EvaluationAndEnhancement")
      ),
    },
  ];

  const location = useLocation();
  const module = 10;
  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 10: Generative Models"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about generative models and their
              applications in AI.
            </p>
          </Row>
          <Row>
            <Col>
              <p>Last Updated: {"2024-06-07"}</p>
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

export default CourseGenerativeModels;
