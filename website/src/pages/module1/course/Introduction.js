import React from "react";
import { Container } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container>
      <h2>Introduction</h2>
      <p>
        Welcome to the Git course! In this course, you will learn how to use Git
        for version control and GitHub for collaboration.
      </p>
      <p>
        Git is a distributed version control system that allows you to track
        changes to files and collaborate with others on projects. GitHub is a
        web-based platform that provides hosting for Git repositories and
        additional collaboration features.
      </p>
      <p>
        By the end of this course, you will be able to use Git to create and
        manage your own repositories, make changes to files, stage and commit
        changes, view the commit history, undo changes, work with branches,
        merge branches, resolve merge conflicts, and collaborate with others
        using GitHub.
      </p>
    </Container>
  );
};

export default Introduction;
