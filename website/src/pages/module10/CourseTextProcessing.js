import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseTextProcessing = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction to NLP",
      component: lazy(() => import("pages/module10/course/Introduction")),
      subLinks: [
        { id: "history", label: "Historical Context" },
        { id: "applications", label: "NLP Applications and Tasks" },
        { id: "challenges", label: "NLP Challenges" },
        { id: "libraries", label: "NLP Libraries" },
      ],
    },
    {
      to: "/preprocessing",
      label: "Text Numerical Representation",
      component: lazy(() => import("pages/module10/course/TextNumericalRepresentation")),
      subLinks: [
        { id: "cleaning", label: "Text Cleaning Techniques" },
        { id: "tokenization", label: "Tokenization Approaches" },
        { id: "tokenizers", label: "Tokenizer Comparison" },
        { id: "special-tokens", label: "Special Tokens" },
        { id: "sequence-length", label: "Sequence Length Handling" },
      ],
    },
    {
      to: "/rnn",
      label: "Recurrent Neural Networks",
      component: lazy(() => import("pages/module10/course/RNN")),
      subLinks: [
      ],
    },
    {
      to: "/transformer-components",
      label: "Transformer Components",
      component: lazy(() => import("pages/module10/course/TransformerComponents")),
      subLinks: [

      ],
    },
    {
      to: "/transformer-architectures",
      label: "Transformer Architectures",
      component: lazy(() => import("pages/module10/course/TransformerArchitectures")),
      subLinks: [
      ],
    },
    {
      to: "/transfer-learning",
      label: "Transfer Learning in NLP",
      component: lazy(() => import("pages/module10/course/TransferLearning")),
      subLinks: [

      ],
    },
    {
      to: "/nlp-evaluation",
      label: "NLP Evaluation",
      component: lazy(() => import("pages/module10/course/NLPEvaluation")),
      subLinks: [

      ],
    },

    {
      to: "/rag",
      label: "Retrieval-Augmented Generation (RAG)",
      component: lazy(() => import("pages/module10/course/RAG")),
      subLinks: [
        { id: "architecture", label: "RAG Architecture and Workflow" },
        { id: "vector-databases", label: "Vector Databases" },
        { id: "chunking", label: "Document Chunking and Indexing" },
        { id: "similarity-search", label: "Similarity Search Algorithms" },
        { id: "pipeline", label: "Complete RAG Pipeline" },
        { id: "evaluation", label: "RAG System Evaluation" },
        { id: "prompt-techniques", label: "Prompt Techniques" },
      ],
    },
    // {
    //   to: "/casestudy",
    //   label: "Case Study",
    //   component: lazy(() => import("pages/module10/course/CaseStudy")),
    //   subLinks: [
    //     { id: "problem", label: "Problem Definition" },
    //     { id: "pipeline", label: "NLP Pipeline" },
    //     { id: "model-selection", label: "Model Selection" },
    //     { id: "hyperparameter", label: "Hyperparameter Optimization" },
    //     { id: "error-analysis", label: "Error Analysis" },
    //     { id: "visualization", label: "Model Outputs and Performance" },
    //   ],
    // },
  ];

  const location = useLocation();
  const module = 10;
  return (
    <ModuleFrame
      module={10}
      isCourse={true}
      title="Module 10: Text Processing and Natural Language Processing"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              This module covers advanced topics in Natural Language Processing
              (NLP) using deep learning techniques. You'll learn about modern NLP
              architectures, from RNNs to Transformers, and gain hands-on experience
              with state-of-the-art models like BERT and GPT. We'll explore text
              preprocessing, word embeddings, transfer learning, and retrieval-augmented
              generation while developing practical skills with PyTorch and the
              Hugging Face ecosystem.
            </p>
          </Row>
          <Row>
            <Col>
              <p>Last Updated: {"2025-03-06"}</p>
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

export default CourseTextProcessing;