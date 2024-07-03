import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseRecommendationSystems = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Recommendation Systems",
      component: lazy(() => import("pages/module11/course/Introduction")),
    },
    {
      to: "/content-based-filtering",
      label: "Content-Based Filtering",
      component: lazy(() =>
        import("pages/module11/course/ContentBasedFiltering")
      ),
    },
    {
      to: "/collaborative-filtering",
      label: "Collaborative Filtering",
      component: lazy(() =>
        import("pages/module11/course/CollaborativeFiltering")
      ),
    },
    {
      to: "/evaluating-recommendation-systems",
      label: "Evaluating Recommendation Systems",
      component: lazy(() =>
        import("pages/module11/course/EvaluatingRecommendationSystems")
      ),
    },
    {
      to: "/scalability-and-challenges",
      label: "Scalability and Real-World Challenges",
      component: lazy(() =>
        import("pages/module11/course/ScalabilityAndChallenges")
      ),
    },
    {
      to: "/advanced-techniques",
      label: "Advanced Recommendation Techniques",
      component: lazy(() => import("pages/module11/course/AdvancedTechniques")),
    },
    {
      to: "/recommendation-systems-in-practice",
      label: "Recommendation Systems in Practice",
      component: lazy(() =>
        import("pages/module11/course/RecommendationSystemsInPractice")
      ),
    },
  ];

  const location = useLocation();
  const module = 11;
  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 11: Recommendation Systems"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about recommendation systems and
              their applications in data science.
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

export default CourseRecommendationSystems;
