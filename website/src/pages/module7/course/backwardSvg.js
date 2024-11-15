const ReverseModeSvg = () => (
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300">
  {/* <!-- Styles --> */}
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>
  
  {/* <!-- Leaf Nodes (Input Variables) --> */}
  {/* <!-- x node --> */}
  <circle cx="100" cy="100" r="30" fill="#e9ecef" stroke="#343a40" stroke-width="2"/>
  <text x="100" cy="100" text-anchor="middle" dominant-baseline="middle" fill="#343a40" font-family="monospace">
    <tspan x="100" dy="-5">x</tspan>
    <tspan x="100" dy="20" font-size="12">[2.0]</tspan>
  </text>
  
  {/* <!-- y node --> */}
  <circle cx="100" cy="200" r="30" fill="#e9ecef" stroke="#343a40" stroke-width="2"/>
  <text x="100" cy="200" text-anchor="middle" dominant-baseline="middle" fill="#343a40" font-family="monospace">
    <tspan x="100" dy="-5">y</tspan>
    <tspan x="100" dy="20" font-size="12">[3.0]</tspan>
  </text>
  
  {/* <!-- Operation Nodes --> */}
  {/* <!-- Multiplication node --> */}
  <rect x="220" y="130" width="60" height="40" rx="5" fill="#228be6" stroke="#1971c2" stroke-width="2"/>
  <text x="250" y="155" text-anchor="middle" dominant-baseline="middle" fill="white" font-family="monospace">*</text>
  
  {/* <!-- Addition node --> */}
  <rect x="380" y="130" width="60" height="40" rx="5" fill="#228be6" stroke="#1971c2" stroke-width="2"/>
  <text x="410" y="155" text-anchor="middle" dominant-baseline="middle" fill="white" font-family="monospace">+</text>
  
  {/* <!-- Result Nodes --> */}
  {/* <!-- z node --> */}
  <circle cx="340" cy="150" r="25" fill="#fab005" stroke="#f08c00" stroke-width="2"/>
  <text x="340" y="150" text-anchor="middle" dominant-baseline="middle" fill="#343a40" font-family="monospace">z</text>
  
  {/* <!-- w node --> */}
  <circle cx="500" cy="150" r="25" fill="#fab005" stroke="#f08c00" stroke-width="2"/>
  <text x="500" y="150" text-anchor="middle" dominant-baseline="middle" fill="#343a40" font-family="monospace">w</text>
  
  {/* <!-- Edges --> */}
  {/* <!-- x to multiplication --> */}
  <line x1="130" y1="100" x2="220" y2="150" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  {/* <!-- y to multiplication --> */}
  <line x1="130" y1="200" x2="220" y2="150" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  {/* <!-- multiplication to z --> */}
  <line x1="280" y1="150" x2="315" y2="150" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  {/* <!-- z to addition --> */}
  <line x1="365" y1="150" x2="380" y2="150" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  {/* <!-- x to addition (direct) --> */}
  <line x1="130" y1="100" x2="380" y2="150" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  {/* <!-- addition to w --> */}
  <line x1="440" y1="150" x2="475" y2="150" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  {/* <!-- Labels --> */}
  <text x="100" y="50" text-anchor="middle" fill="#495057" font-size="14">Leaf Nodes</text>
  <text x="250" y="100" text-anchor="middle" fill="#495057" font-size="14">Operations</text>
  <text x="500" y="50" text-anchor="middle" fill="#495057" font-size="14">Result</text>

  {/* <!-- Legend --> */}
  <rect x="20" y="250" width="560" height="40" rx="5" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
  <circle cx="50" cy="270" r="10" fill="#e9ecef" stroke="#343a40" stroke-width="2"/>
  <text x="70" y="270" dominant-baseline="middle" fill="#495057" font-size="12">Input Variables</text>
  <rect x="150" y="260" width="20" height="20" rx="2" fill="#228be6" stroke="#1971c2" stroke-width="2"/>
  <text x="180" y="270" dominant-baseline="middle" fill="#495057" font-size="12">Operations</text>
  <circle cx="270" cy="270" r="10" fill="#fab005" stroke="#f08c00" stroke-width="2"/>
  <text x="290" y="270" dominant-baseline="middle" fill="#495057" font-size="12">Intermediate Values</text>
  <line x1="370" y1="270" x2="410" y2="270" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <text x="440" y="270" dominant-baseline="middle" fill="#495057" font-size="12">Data Flow</text>
</svg>
);

export default ReverseModeSvg;