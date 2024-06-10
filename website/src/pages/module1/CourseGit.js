import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";

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
      to: "/first-steps-with-git",
      label: "First Steps with Git",
      component: lazy(() => import("pages/module1/course/FirstStepsWithGit")),
      subLinks: [
        { id: "creating-repository", label: "Creating a Git repository" },
        { id: "staging-changes", label: "Staging Changes" },
        { id: "committing-changes", label: "Committing Changes" },
        { id: "viewing-commit-history", label: "Viewing the Commit History" },
        { id: "undoing-changes", label: "Undoing Changes" },
      ],
    },
    {
      to: "/working-with-remote-repositories",
      label: "Working with Remote Repositories",
      component: lazy(() =>
        import("pages/module1/course/WorkingWithRemoteRepositories")
      ),
      subLinks: [
        { id: "create-github-account", label: "Create GitHub Account" },
        {
          id: "configure-and-access-github",
          label: "Configure and Access GitHub",
        },
        {
          id: "managing-remote-repositories",
          label: "Managing Remote Repositories",
        },
      ],
    },
    // {
    //   to: "/branching-and-merging",
    //   label: "Branching and Merging",
    //   component: lazy(() => import("pages/module1/course/BranchingAndMerging")),
    //   subLinks: [
    //     { id: "working-with-branches", label: "Working with Branches" },
    //     { id: "merging-branches", label: "Merging Branches" },
    //     { id: "resolving-merge-conflicts", label: "Resolving Merge Conflicts" },
    //   ],
    // },
    // {
    //   to: "/collaborating",
    //   label: "Collaborating",
    //   component: lazy(() => import("pages/module1/course/Collaborating")),
    //   subLinks: [
    //     {
    //       id: "collaborating-with-others-using-github",
    //       label: "Using GitHub for Collaboration",
    //     },
    //     { id: "git-workflows", label: "Git Workflows" },
    //     { id: "peer-reviews", label: "Peer Reviews" },
    //   ],
    // },
    // {
    //   to: "/best-practices-and-resources",
    //   label: "Best Practices and Resources",
    //   component: lazy(() =>
    //     import("pages/module1/course/BestPracticesAndResources")
    //   ),
    // },
  ];

  return (
    <ModuleFrame
      module={1}
      isCourse={true}
      title="Module 1: Git"
      courseLinks={courseLinks}
    >
      <Row>
        <p>
          Learn how to use Git for version control and GitHub for collaboration.
        </p>
      </Row>
      <Row>
        <Col>
          <p>Last Updated: {"2024-06-07"}</p>
        </Col>
      </Row>
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseGit;
