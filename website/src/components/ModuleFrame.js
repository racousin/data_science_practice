import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import ModuleNavigation from "components/ModuleNavigation";

const ModuleFrame = ({ module, isCourse, title, children }) => {
  return (
    <Row>
      <ModuleNavigation module={module} isCourse={isCourse} title={title} />
      <Col md={12} className="module-content">
        {children}
      </Col>
    </Row>
  );
};

export default ModuleFrame;
