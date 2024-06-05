import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CourseImageProcessing = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Image Processing",
      component: lazy(() => import("pages/module11/course/Introduction")),
    },
    {
      to: "/basic-image-manipulations",
      label: "Basic Image Manipulations",
      component: lazy(() =>
        import("pages/module11/course/BasicImageManipulations")
      ),
    },
    {
      to: "/image-enhancement-techniques",
      label: "Image Enhancement Techniques",
      component: lazy(() =>
        import("pages/module11/course/ImageEnhancementTechniques")
      ),
    },
    {
      to: "/feature-detection-description",
      label: "Feature Detection and Description",
      component: lazy(() =>
        import("pages/module11/course/FeatureDetectionDescription")
      ),
    },
    {
      to: "/image-segmentation",
      label: "Image Segmentation",
      component: lazy(() => import("pages/module11/course/ImageSegmentation")),
    },
    {
      to: "/object-detection-recognition",
      label: "Object Detection and Recognition",
      component: lazy(() =>
        import("pages/module11/course/ObjectDetectionRecognition")
      ),
    },
    {
      to: "/advanced-applications-techniques",
      label: "Advanced Applications and Techniques",
      component: lazy(() =>
        import("pages/module11/course/AdvancedApplicationsTechniques")
      ),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation
          module={11}
          isCourse={true}
          title="Module 11: Image Processing"
        />
      </Row>
      <Row>
        <p>
          In this module, you will learn about image processing techniques and
          applications.
        </p>
      </Row>
      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module11/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseImageProcessing;
