import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CourseRecommendationSystems = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Recommendation Systems",
      component: lazy(() => import("pages/module13/course/Introduction")),
    },
    {
      to: "/content-based-filtering",
      label: "Content-Based Filtering",
      component: lazy(() =>
        import("pages/module13/course/ContentBasedFiltering")
      ),
    },
    {
      to: "/collaborative-filtering",
      label: "Collaborative Filtering",
      component: lazy(() =>
        import("pages/module13/course/CollaborativeFiltering")
      ),
    },
    {
      to: "/evaluating-recommendation-systems",
      label: "Evaluating Recommendation Systems",
      component: lazy(() =>
        import("pages/module13/course/EvaluatingRecommendationSystems")
      ),
    },
    {
      to: "/scalability-and-challenges",
      label: "Scalability and Real-World Challenges",
      component: lazy(() =>
        import("pages/module13/course/ScalabilityAndChallenges")
      ),
    },
    {
      to: "/advanced-techniques",
      label: "Advanced Recommendation Techniques",
      component: lazy(() => import("pages/module13/course/AdvancedTechniques")),
    },
    {
      to: "/recommendation-systems-in-practice",
      label: "Recommendation Systems in Practice",
      component: lazy(() =>
        import("pages/module13/course/RecommendationSystemsInPractice")
      ),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={13}
          isCourse={true}
          title="Module 13: Recommendation Systems"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about recommendation systems and their
          applications in data science.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module13/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseRecommendationSystems;
