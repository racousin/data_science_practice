import React from "react";
import { Container, Row, Col, Nav } from "react-bootstrap";
import { Routes, Route, Link } from "react-router-dom";
import Introduction from "pages/module1/course/Introduction";
import InstallingGit from "pages/module1/course/InstallingGit";
import CreatingRepository from "pages/module1/course/CreatingRepository";
import MakingChangesToFiles from "pages/module1/course/MakingChangesToFiles";
import StagingChanges from "pages/module1/course/StagingChanges";
import CommittingChanges from "pages/module1/course/CommittingChanges";
import ViewingTheCommitHistory from "pages/module1/course/ViewingTheCommitHistory";
import UndoingChanges from "pages/module1/course/UndoingChanges";
import WorkingWithBranches from "pages/module1/course/WorkingWithBranches";
import MergingBranches from "pages/module1/course/MergingBranches";
import ResolvingMergeConflicts from "pages/module1/course/ResolvingMergeConflicts";
import WorkingWithRemoteRepositories from "pages/module1/course/WorkingWithRemoteRepositories";
import CollaboratingWithOthersUsingGitHub from "pages/module1/course/CollaboratingWithOthersUsingGitHub";
import BestPracticesForUsingGit from "pages/module1/course/BestPracticesForUsingGit";

const CourseGit = () => {
  return (
    <Container>
      <h1 className="my-4">Module 1: Git</h1>
      <p>
        In this module, students will learn how to use Git for version control
        and GitHub for collaboration.
      </p>
      <Row>
        <Col md={3}>
          <Nav variant="pills" className="flex-column">
            <Nav.Link as={Link} to="/module1/course/introduction">
              Introduction
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/installing-git">
              Installing Git
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/creating-repository">
              Creating a Git repository
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/making-changes-to-files">
              Making changes to files
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/staging-changes">
              Staging changes
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/committing-changes">
              Committing changes
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/viewing-the-commit-history">
              Viewing the commit history
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/undoing-changes">
              Undoing changes
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/working-with-branches">
              Working with branches
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/merging-branches">
              Merging branches
            </Nav.Link>
            <Nav.Link as={Link} to="/module1/course/resolving-merge-conflicts">
              Resolving merge conflicts
            </Nav.Link>
            <Nav.Link
              as={Link}
              to="/module1/course/working-with-remote-repositories"
            >
              Working with remote repositories
            </Nav.Link>
            <Nav.Link
              as={Link}
              to="/module1/course/collaborating-with-others-using-github"
            >
              Collaborating with others using GitHub
            </Nav.Link>
            <Nav.Link
              as={Link}
              to="/module1/course/best-practices-for-using-git"
            >
              Best practices for using Git
            </Nav.Link>
          </Nav>
        </Col>
        <Col md={9}>
          <Routes>
            {" "}
            <Route path="introduction" element={<Introduction />} />
            <Route path="installing-git" element={<InstallingGit />} />
            <Route
              path="creating-repository"
              element={<CreatingRepository />}
            />
            <Route
              path="making-changes-to-files"
              element={<MakingChangesToFiles />}
            />
            <Route path="staging-changes" element={<StagingChanges />} />
            <Route path="committing-changes" element={<CommittingChanges />} />
            <Route
              path="viewing-the-commit-history"
              element={<ViewingTheCommitHistory />}
            />
            <Route path="undoing-changes" element={<UndoingChanges />} />
            <Route
              path="working-with-branches"
              element={<WorkingWithBranches />}
            />
            <Route path="merging-branches" element={<MergingBranches />} />
            <Route
              path="resolving-merge-conflicts"
              element={<ResolvingMergeConflicts />}
            />
            <Route
              path="working-with-remote-repositories"
              element={<WorkingWithRemoteRepositories />}
            />
            <Route
              path="collaborating-with-others-using-github"
              element={<CollaboratingWithOthersUsingGitHub />}
            />
            <Route
              path="best-practices-for-using-git"
              element={<BestPracticesForUsingGit />}
            />
          </Routes>
        </Col>
      </Row>
    </Container>
  );
};

export default CourseGit;
