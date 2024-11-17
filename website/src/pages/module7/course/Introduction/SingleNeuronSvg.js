import React from 'react';

const SingleNeuronSvg = () => {
  return (
    <svg viewBox="0 0 600 300" xmlns="http://www.w3.org/2000/svg">
      {/* Background */}
      <rect width="600" height="300" fill="white"/>
      
      {/* Input connections */}
      <g stroke="#999" strokeWidth="1.5" opacity="0.6">
        <line x1="100" y1="80" x2="250" y2="150" />
        <line x1="100" y1="150" x2="250" y2="150" />
        <line x1="100" y1="220" x2="250" y2="150" />
      </g>
      
      {/* Input labels */}
      <text x="50" y="85" fontSize="14" fill="#666">x₁</text>
      <text x="50" y="155" fontSize="14" fill="#666">x₂</text>
      <text x="50" y="225" fontSize="14" fill="#666">x₃</text>
      
      {/* Weight labels */}
      <text x="160" y="85" fontSize="14" fill="#666">w₁</text>
      <text x="160" y="135" fontSize="14" fill="#666">w₂</text>
      <text x="160" y="185" fontSize="14" fill="#666">w₃</text>
      
      {/* Input nodes */}
      <circle cx="80" cy="80" r="15" fill="#6495ED"/>
      <circle cx="80" cy="150" r="15" fill="#6495ED"/>
      <circle cx="80" cy="220" r="15" fill="#6495ED"/>
      
      {/* Summation node */}
      <circle cx="250" cy="150" r="25" fill="#4CAF50"/>
      <text x="240" y="155" fontSize="20" fill="white">Σ</text>
      
      {/* Activation function */}
      <path d="M 300 150 Q 350 150 350 100 Q 350 50 400 50" 
            fill="none" stroke="#FFA500" strokeWidth="2"/>
      <text x="340" y="40" fontSize="14" fill="#666">g(z)</text>
      
      {/* Output */}
      <circle cx="450" cy="150" r="15" fill="#FFA500"/>
      <text x="480" y="155" fontSize="14" fill="#666">y = g(Σwᵢxᵢ + b)</text>
      
      {/* Bias */}
      <line x1="250" y1="80" x2="250" y2="125" stroke="#999" strokeWidth="1.5" opacity="0.6"/>
      <text x="240" y="70" fontSize="14" fill="#666">b</text>
      
      {/* Connection to output */}
      <line x1="275" y1="150" x2="435" y2="150" stroke="#999" strokeWidth="1.5" opacity="0.6"/>

      {/* Mathematical expressions */}
      <text x="100" y="270" fontSize="14" fill="#666">Input: xᵢ ∈ ℝⁿ</text>
      <text x="250" y="270" fontSize="14" fill="#666">a = Σwᵢxᵢ + b</text>
      <text x="400" y="270" fontSize="14" fill="#666">Output: y = g(a)</text>
    </svg>
  );
};

export default SingleNeuronSvg;