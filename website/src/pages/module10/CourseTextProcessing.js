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
      label: "TextNumericalRepresentation",
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
      label: "Recurrent Neural Networks for NLP",
      component: lazy(() => import("pages/module10/course/RNN")),
      subLinks: [
        { id: "sequential-data", label: "Sequential Nature of Text Data" },
        { id: "basic-rnn", label: "Basic RNN Architecture" },
        { id: "lstm-gru", label: "LSTM and GRU Architectures" },
        { id: "vanishing-gradient", label: "Vanishing/Exploding Gradient Problem" },
        { id: "bidirectional", label: "Bidirectional RNNs" },
        { id: "implementations", label: "PyTorch Implementations" },
        { id: "comparison", label: "Comparison with Transformers" },
      ],
    },
    {
      to: "/transformer",
      label: "Transformer Architecture",
      component: lazy(() => import("pages/module10/course/Transformer")),
      subLinks: [
        { id: "architecture", label: "Transformer Model Architecture" },
        { id: "self-attention", label: "Self-Attention Mechanism" },
        { id: "multi-head-attention", label: "Multi-Head Attention" },
        { id: "positional-encodings", label: "Positional Encodings" },
        { id: "encoder-decoder", label: "Encoder and Decoder Components" },
        { id: "attention-math", label: "Mathematical Formulation of Attention" },
        { id: "information-flow", label: "Information Flow" },
      ],
    },
    {
      to: "/TransformerArchitectures",
      label: "TransformerArchitectures",
      component: lazy(() => import("pages/module10/course/TransformerArchitectures")),
      subLinks: [
        { id: "architecture", label: "BERT Architecture and Innovations" },
        { id: "pretraining", label: "Masked Language Modeling and NSP" },
        { id: "classification", label: "Classification and Token-level Tasks" },
        { id: "variants", label: "BERT Variants" },
        { id: "fine-tuning", label: "Fine-tuning Techniques" },
        { id: "bidirectional-context", label: "Bidirectional Context" },
      ],
    },
    // {
    //   to: "/transfer-learning",
    //   label: "Transfer Learning in NLP",
    //   component: lazy(() => import("pages/module10/course/TransferLearning")),
    //   subLinks: [
    //     { id: "concept", label: "Concept and Importance" },
    //     { id: "pre-training", label: "Pre-training and Fine-tuning" },
    //     { id: "strategies", label: "Transfer Learning Strategies" },
    //     { id: "domain-adaptation", label: "Domain Adaptation" },
    //     { id: "low-resource", label: "Low-Resource Fine-tuning" },
    //     { id: "parameter-efficient", label: "Parameter-Efficient Methods" },
    //     { id: "frozen-vs-trainable", label: "Frozen vs. Trainable Parameters" },
    //     { id: "guidelines", label: "When to Use Different Methods" },
    //   ],
    // },

    // {
    //   to: "/huggingface",
    //   label: "Hugging Face Ecosystem",
    //   component: lazy(() => import("pages/module10/course/HuggingFace")),
    //   subLinks: [
    //     { id: "hub", label: "Hugging Face Hub" },
    //     { id: "transformers", label: "Transformers Library" },
    //     { id: "models-tokenizers", label: "Loading Models and Tokenizers" },
    //     { id: "trainer", label: "Fine-tuning with Trainer API" },
    //     { id: "datasets", label: "Datasets and Evaluation" },
    //     { id: "sharing", label: "Model Sharing and Collaboration" },
    //   ],
    // },
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