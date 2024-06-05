import React from "react";
import { Nav, Button } from "react-bootstrap";
import { useNavigate } from "react-router-dom";
import EvaluationModal from "components/EvaluationModal";

const ModuleNavigation = ({ module, isCourse, title = "" }) => {
  const navigate = useNavigate();

  const navigateTo = (path) => {
    navigate(path);
  };

  return (
    <Nav className="justify-content-between align-items-center navigation-header">
      <h1 className="module-title">{title}</h1>
      <div>
        {module > 1 && (
          <Button
            variant="outline-primary"
            className="nav-button button-outline"
            onClick={() => navigateTo(`/module${module - 1}/course`)}
          >
            Previous Module
          </Button>
        )}
        {isCourse ? (
          <Button
            variant="outline-secondary"
            className="nav-button button-outline"
            onClick={() => navigateTo(`/module${module}/exercise`)}
          >
            Exercises
          </Button>
        ) : (
          <>
            <Button
              variant="outline-secondary"
              className="nav-button button-outline"
              onClick={() => navigateTo(`/module${module}/course`)}
            >
              Courses
            </Button>
            <EvaluationModal />
          </>
        )}

        {module < 4 && (
          <Button
            variant="outline-success"
            className="nav-button button-outline"
            onClick={() => navigateTo(`/module${module + 1}/course`)}
          >
            Next Module
          </Button>
        )}
      </div>
    </Nav>
  );
};

export default ModuleNavigation;
