import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleNavigation from "components/ModuleNavigation";

const CourseGit = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction",
      component: lazy(() => import("pages/module1/course/Introduction")),
    },
    {
      to: "/installing-git",
      label: "Installing Git",
      component: lazy(() => import("pages/module1/course/InstallingGit")),
    },
    {
      to: "/creating-repository",
      label: "Creating a Git repository",
      component: lazy(() => import("pages/module1/course/CreatingRepository")),
    },
    {
      to: "/making-changes-to-files",
      label: "Making changes to files",
      component: lazy(() =>
        import("pages/module1/course/MakingChangesToFiles")
      ),
    },
    {
      to: "/staging-changes",
      label: "Staging changes",
      component: lazy(() => import("pages/module1/course/StagingChanges")),
    },
    {
      to: "/committing-changes",
      label: "Committing changes",
      component: lazy(() => import("pages/module1/course/CommittingChanges")),
    },
    {
      to: "/viewing-the-commit-history",
      label: "Viewing the commit history",
      component: lazy(() =>
        import("pages/module1/course/ViewingTheCommitHistory")
      ),
    },
    {
      to: "/undoing-changes",
      label: "Undoing changes",
      component: lazy(() => import("pages/module1/course/UndoingChanges")),
    },
    {
      to: "/working-with-branches",
      label: "Working with branches",
      component: lazy(() => import("pages/module1/course/WorkingWithBranches")),
    },
    {
      to: "/merging-branches",
      label: "Merging branches",
      component: lazy(() => import("pages/module1/course/MergingBranches")),
    },
    {
      to: "/resolving-merge-conflicts",
      label: "Resolving merge conflicts",
      component: lazy(() =>
        import("pages/module1/course/ResolvingMergeConflicts")
      ),
    },
    {
      to: "/working-with-remote-repositories",
      label: "Working with remote repositories",
      component: lazy(() =>
        import("pages/module1/course/WorkingWithRemoteRepositories")
      ),
    },
    {
      to: "/collaborating-with-others-using-github",
      label: "Collaborating with others using GitHub",
      component: lazy(() =>
        import("pages/module1/course/CollaboratingWithOthersUsingGitHub")
      ),
    },
    {
      to: "/best-practices-for-using-git",
      label: "Best practices for using Git",
      component: lazy(() =>
        import("pages/module1/course/BestPracticesForUsingGit")
      ),
    },
  ];

  return (
    <Container fluid>
      <Row>
        <ModuleNavigation module={1} isCourse={true} title="Module 1: Git" />
      </Row>
      <Row>
        <p>
          In this module, students will learn how to use Git for version control
          and GitHub for collaboration.
        </p>
      </Row>

      <Row>
        <Col md={3}>
          <NavigationMenu links={courseLinks} prefix={"/module1/course"} />
        </Col>
        <Col md={9}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </Container>
  );
};

export default CourseGit;
