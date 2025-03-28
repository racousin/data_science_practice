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

        { id: "text-representation", label: "Text Data Representation" },
        { id: "history", label: "Historical Context" },
        { id: "applications", label: "NLP Applications and Tasks" }
      ],
    },
    {
      to: "/preprocessing",
      label: "Text Numerical Representation",
      component: lazy(() => import("pages/module10/course/TextNumericalRepresentation")),
      subLinks: [
        { id: "tokenization", label: "Tokenization" },
        { id: "word-tokenization", label: "Word-Level Tokenization" },
        { id: "character-tokenization", label: "Character-Level Tokenization" },
        { id: "subword-tokenization", label: "Subword Tokenization" },
        { id: "embeddings", label: "Embeddings" },
        { id: "learned-embeddings", label: "Learned Embeddings" },
        { id: "contextual-embeddings", label: "Contextual Embeddings" },
      ],
    },
    {
      to: "/rnn",
      label: "Recurrent Neural Networks",
      component: lazy(() => import("pages/module10/course/RNN")),
      subLinks: [
        {id:"rnn-introduction", label:"Introduction to RNNs"},
        {id:"rnn-notation", label:"RNN Notation"},
        {id:"units-architecture", label:"Units Architecture"},
        {id:"backpropagation-through-time", label:"Backpropagation Through Time (BPTT)"},
        {id:"torch-example-prediction", label:"Torch Example Sequence Prediction"}
      ],
    },
    {
      to: "/transformer-components",
      label: "Transformer Components",
      component: lazy(() => import("pages/module10/course/TransformerComponents")),
      subLinks: [
        {id:"transformer-introduction", label:"Introduction to Transformers"},
        {id:"transformer-notation", label:"Transformer Notation"},
        {id:"transformer-components-layers", label:"Transformers Components Layers"},
        {id:"attention-backprop", label:"Backpropagation Through Attention"},
        {id:"minimal-transformer-implementation", label:"Minimal Transformer Implementation"}

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

    // {
    //   to: "/rag",
    //   label: "Retrieval-Augmented Generation (RAG)",
    //   component: lazy(() => import("pages/module10/course/RAG")),
    //   subLinks: [
    //     { id: "architecture", label: "RAG Architecture and Workflow" },
    //     { id: "vector-databases", label: "Vector Databases" },
    //     { id: "chunking", label: "Document Chunking and Indexing" },
    //     { id: "similarity-search", label: "Similarity Search Algorithms" },
    //     { id: "pipeline", label: "Complete RAG Pipeline" },
    //     { id: "evaluation", label: "RAG System Evaluation" },
    //     { id: "prompt-techniques", label: "Prompt Techniques" },
    //   ],
    // },
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
      title="Module 10: Natural Language Processing"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <Col>
              <p>Last Updated: {"2025-03-27"}</p>
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