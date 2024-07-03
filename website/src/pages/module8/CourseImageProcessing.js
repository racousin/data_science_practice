import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseImageProcessing = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to Image Processing",
      component: lazy(() => import("pages/module8/course/Introduction")),
    },
    {
      to: "/basic-image-manipulations",
      label: "Basic Image Manipulations",
      component: lazy(() =>
        import("pages/module8/course/BasicImageManipulations")
      ),
    },
    {
      to: "/image-enhancement-techniques",
      label: "Image Enhancement Techniques",
      component: lazy(() =>
        import("pages/module8/course/ImageEnhancementTechniques")
      ),
    },
    {
      to: "/feature-detection-description",
      label: "Feature Detection and Description",
      component: lazy(() =>
        import("pages/module8/course/FeatureDetectionDescription")
      ),
    },
    {
      to: "/image-segmentation",
      label: "Image Segmentation",
      component: lazy(() => import("pages/module8/course/ImageSegmentation")),
    },
    {
      to: "/object-detection-recognition",
      label: "Object Detection and Recognition",
      component: lazy(() =>
        import("pages/module8/course/ObjectDetectionRecognition")
      ),
    },
    {
      to: "/advanced-applications-techniques",
      label: "Advanced Applications and Techniques",
      component: lazy(() =>
        import("pages/module8/course/AdvancedApplicationsTechniques")
      ),
    },
  ];

  const location = useLocation();
  const module = 8;
  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 8: Image Processing"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about image processing techniques
              and applications.
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

export default CourseImageProcessing;
