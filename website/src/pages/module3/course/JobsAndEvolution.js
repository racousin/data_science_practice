import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const JobsAndEvolution = () => {
  return (
    <Container>
      <h1 className="my-4">Jobs and Evolution</h1>
      <p>
        In this section, you will learn about the jobs and evolution of data
        science.
      </p>
      <Row>
        <Col>
          <h2>Data Science Jobs</h2>
          <p>
            Data science jobs have become increasingly popular in recent years,
            with a growing demand for professionals who can analyze and
            interpret data to drive business value. Some common data science
            jobs include:
          </p>
          <ul>
            <li>Data Scientist</li>
            <li>Data Engineer</li>
            <li>Data Analyst</li>
            <li>Machine Learning Engineer</li>
            <li>Business Intelligence Analyst</li>
          </ul>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Evolution of Data Science</h2>
          <p>
            Data science has evolved from a niche field to a mainstream
            discipline, with a growing number of organizations recognizing the
            value of data-driven decision making. This evolution has been driven
            by several factors, including:
          </p>
          <ul>
            <li>The increasing availability of data</li>
            <li>
              The development of new technologies for data storage, processing,
              and analysis
            </li>
            <li>
              The growing demand for data-driven insights to improve business
              outcomes
            </li>
          </ul>
        </Col>
      </Row>
    </Container>
  );
};

export default JobsAndEvolution;
