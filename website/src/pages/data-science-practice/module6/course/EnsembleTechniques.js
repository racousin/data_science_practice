import React from 'react';
import { Container, Title, Text, Stack, List, Group, Table } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

// TODO add boosting

const BaggingSVG = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
    <rect x="10" y="10" width="120" height="80" fill="#f0f0f0" stroke="#000000" strokeWidth="2"/>
    <text x="70" y="55" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Original Dataset</text>

    <rect x="200" y="10" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Bootstrap Sample 1</text>

    <rect x="200" y="80" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Bootstrap Sample 2</text>

    <rect x="200" y="150" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Bootstrap Sample 3</text>

    <rect x="400" y="10" width="80" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="440" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Model 1</text>

    <rect x="400" y="80" width="80" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="440" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Model 2</text>

    <rect x="400" y="150" width="80" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="440" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Model 3</text>

    <rect x="560" y="10" width="80" height="60" fill="#e6ffe6" stroke="#000000" strokeWidth="2"/>
    <text x="600" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Prediction 1</text>

    <rect x="560" y="80" width="80" height="60" fill="#e6ffe6" stroke="#000000" strokeWidth="2"/>
    <text x="600" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Prediction 2</text>

    <rect x="560" y="150" width="80" height="60" fill="#e6ffe6" stroke="#000000" strokeWidth="2"/>
    <text x="600" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Prediction 3</text>

    <rect x="700" y="80" width="90" height="60" fill="#ffffe6" stroke="#000000" strokeWidth="2"/>
    <text x="745" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Final Prediction</text>

    <defs>
      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" />
      </marker>
    </defs>

    <line x1="130" y1="50" x2="190" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="130" y1="50" x2="190" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="130" y1="50" x2="190" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    <line x1="300" y1="40" x2="390" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="300" y1="110" x2="390" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="300" y1="180" x2="390" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    <line x1="480" y1="40" x2="550" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="480" y1="110" x2="550" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="480" y1="180" x2="550" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    <line x1="640" y1="40" x2="690" y2="100" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="640" y1="110" x2="690" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="640" y1="180" x2="690" y2="120" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    <text x="160" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Bootstrap</text>
    <text x="350" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Train</text>
    <text x="520" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predict</text>
    <text x="670" y="70" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Aggregate</text>
  </svg>
);

const StackingSVG = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
    {/* Original Dataset */}
    <rect x="10" y="10" width="120" height="80" fill="#f0f0f0" stroke="#000000" strokeWidth="2"/>
    <text x="70" y="55" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Original Dataset</text>

    {/* Base Models */}
    <rect x="200" y="10" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Base Model 1</text>

    <rect x="200" y="80" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Base Model 2</text>

    <rect x="200" y="150" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Base Model 3</text>

    {/* Predictions / Meta-features */}
    <rect x="400" y="10" width="100" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="450" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predictions 1</text>

    <rect x="400" y="80" width="100" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="450" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predictions 2</text>

    <rect x="400" y="150" width="100" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="450" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predictions 3</text>

    {/* Meta-features */}
    <rect x="560" y="80" width="100" height="60" fill="#e6ffe6" stroke="#000000" strokeWidth="2"/>
    <text x="610" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Meta-features</text>

    {/* Meta-model */}
    <rect x="700" y="80" width="90" height="60" fill="#ffffe6" stroke="#000000" strokeWidth="2"/>
    <text x="745" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Meta-model</text>

    {/* Arrows */}
    <defs>
      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" />
      </marker>
    </defs>

    {/* Original Dataset to Base Models */}
    <line x1="130" y1="50" x2="190" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="130" y1="50" x2="190" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="130" y1="50" x2="190" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    {/* Base Models to Predictions */}
    <line x1="300" y1="40" x2="390" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="300" y1="110" x2="390" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="300" y1="180" x2="390" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    {/* Predictions to Meta-features */}
    <line x1="500" y1="40" x2="550" y2="100" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="500" y1="110" x2="550" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="500" y1="180" x2="550" y2="120" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    {/* Meta-features to Meta-model */}
    <line x1="660" y1="110" x2="690" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    {/* Labels */}
    <text x="160" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Train</text>
    <text x="350" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predict</text>
    <text x="530" y="70" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Combine</text>
    <text x="680" y="70" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Train</text>
  </svg>
);


const EnsembleTechniques = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Ensemble Techniques</Title>

      <Stack spacing="xl">
        <Title order={3} id="bagging">Bagging (Bootstrap Aggregating)</Title>
<BaggingSVG/>
        <Title order={3} id="stacking">Stacking</Title>
        <StackingSVG/>
      </Stack>
    </Container>
  );
};


export default EnsembleTechniques;