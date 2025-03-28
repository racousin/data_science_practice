const ArchitectureSvg = () => {
  return (
      <svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  {/* <!-- Background --> */}
  <rect width="800" height="400" fill="white"/>
  
  {/* <!-- Layer Labels --> */}
  <text x="50" y="30" font-size="16" fill="#333">Input Layer</text>
  <text x="320" y="30" font-size="16" fill="#333">Hidden Layer</text>
  <text x="600" y="30" font-size="16" fill="#333">Output Layer</text>
  
  {/* <!-- Input Layer Neurons --> */}
  <circle cx="100" cy="100" r="20" fill="#6495ED" />
  <circle cx="100" cy="200" r="20" fill="#6495ED" />
  <circle cx="100" cy="300" r="20" fill="#6495ED" />
  
  {/* <!-- Hidden Layer Neurons --> */}
  <circle cx="350" cy="80" r="20" fill="#4CAF50" />
  <circle cx="350" cy="160" r="20" fill="#4CAF50" />
  <circle cx="350" cy="240" r="20" fill="#4CAF50" />
  <circle cx="350" cy="320" r="20" fill="#4CAF50" />
  
  {/* <!-- Output Layer Neurons --> */}
  <circle cx="600" cy="150" r="20" fill="#FFA500" />
  <circle cx="600" cy="250" r="20" fill="#FFA500" />
  
  {/* <!-- Connections from Input to Hidden Layer --> */}
  <g stroke="#999" stroke-width="1.5" opacity="0.6">
    {/* <!-- From first input neuron --> */}
    <line x1="120" y1="100" x2="330" y2="80" />
    <line x1="120" y1="100" x2="330" y2="160" />
    <line x1="120" y1="100" x2="330" y2="240" />
    <line x1="120" y1="100" x2="330" y2="320" />
    
    {/* <!-- From second input neuron --> */}
    <line x1="120" y1="200" x2="330" y2="80" />
    <line x1="120" y1="200" x2="330" y2="160" />
    <line x1="120" y1="200" x2="330" y2="240" />
    <line x1="120" y1="200" x2="330" y2="320" />
    
    {/* <!-- From third input neuron --> */}
    <line x1="120" y1="300" x2="330" y2="80" />
    <line x1="120" y1="300" x2="330" y2="160" />
    <line x1="120" y1="300" x2="330" y2="240" />
    <line x1="120" y1="300" x2="330" y2="320" />
  </g>
  
  {/* <!-- Connections from Hidden to Output Layer --> */}
  <g stroke="#999" stroke-width="1.5" opacity="0.6">
    {/* <!-- To first output neuron --> */}
    <line x1="370" y1="80" x2="580" y2="150" />
    <line x1="370" y1="160" x2="580" y2="150" />
    <line x1="370" y1="240" x2="580" y2="150" />
    <line x1="370" y1="320" x2="580" y2="150" />
    
    {/* <!-- To second output neuron --> */}
    <line x1="370" y1="80" x2="580" y2="250" />
    <line x1="370" y1="160" x2="580" y2="250" />
    <line x1="370" y1="240" x2="580" y2="250" />
    <line x1="370" y1="320" x2="580" y2="250" />
  </g>
  
  {/* <!-- Layer Annotations --> */}
  <text x="60" y="350" font-size="12" fill="#666">x₁, x₂, x₃</text>
  <text x="320" y="350" font-size="12" fill="#666">o₁, o₂, o₃, o₄</text>
  <text x="580" y="350" font-size="12" fill="#666">y₁, y₂</text>
  
  {/* <!-- Weight Annotation Example -->
  <text x="95" y="100" font-size="12" fill="#666">w¹ᵢⱼ</text>
  <text x="450" y="130" font-size="12" fill="#666">w²ᵢⱼ</text>
   */}
  {/* <!-- Mathematical Expression Examples --> */}
  <text x="350" y="50" font-size="12" fill="#666">o = g(W¹x + b¹)</text>
  <text x="600" y="50" font-size="12" fill="#666">y = g(W²h + b²)</text>
</svg>
  )
}

export default ArchitectureSvg;