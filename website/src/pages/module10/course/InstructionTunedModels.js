import React from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const InstructionTunedModels = () => {
  return (
    <div className="max-w-5xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h1 className="text-3xl font-bold mb-8 text-center">
        The Evolution of Transformer Models: Pre-Training to Instruction Tuning
      </h1>
      
      {/* Training Pipeline Visualization */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Training Pipeline</h2>
        <div className="flex flex-wrap justify-between items-center text-center">
          {[
            { name: "Pre-trained LLM", example: "GPT-3, Llama 2", color: "bg-blue-100" },
            { name: "Supervised Fine-Tuning", example: "InstructGPT", color: "bg-green-100" },
            { name: "RLHF", example: "ChatGPT", color: "bg-yellow-100" },
            { name: "Deployed Assistant", example: "Production System", color: "bg-purple-100" }
          ].map((stage, index) => (
            <React.Fragment key={index}>
              <div className={`${stage.color} p-4 rounded-lg w-48 h-32 flex flex-col justify-center`}>
                <div className="font-bold mb-2">{stage.name}</div>
                <div className="text-sm italic">{stage.example}</div>
              </div>
              {index < 3 && (
                <div className="text-gray-500 text-3xl mx-2">→</div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
      
      {/* Model Comparison Section */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Model Input/Output Patterns</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Base Transformer */}
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="text-xl font-medium mb-3">Base Transformer (GPT, Llama)</h3>
            <p className="mb-3">Trained to predict next token given previous context.</p>
            <div className="bg-gray-50 p-3 rounded mb-3">
              <h4 className="font-medium mb-2">Input:</h4>
              <div className="text-sm font-mono bg-gray-100 p-2 rounded">
                The quick brown fox jumps
              </div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <h4 className="font-medium mb-2">Output:</h4>
              <div className="text-sm font-mono bg-gray-100 p-2 rounded">
                over the lazy dog.
              </div>
            </div>
          </div>
          
          {/* Instruction-Tuned */}
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="text-xl font-medium mb-3">Instruction-Tuned (Llama Chat, Gemma)</h3>
            <p className="mb-3">Fine-tuned to follow natural language instructions.</p>
            <div className="bg-gray-50 p-3 rounded mb-3">
              <h4 className="font-medium mb-2">Input:</h4>
              <div className="text-sm font-mono bg-gray-100 p-2 rounded">
                Write a short poem about artificial intelligence.
              </div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <h4 className="font-medium mb-2">Output:</h4>
              <div className="text-sm font-mono bg-gray-100 p-2 rounded">
                Silicon dreams in neural space,<br/>
                Learning patterns, growing wise.<br/>
                Not human, but with human grace,<br/>
                A mind that learns but never dies.
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Training Stages Detail */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-6">Training Stages Explained</h2>
        
        <div className="mb-8">
          <h3 className="text-xl font-medium mb-3">1. Pre-training</h3>
          <p className="mb-3">
            Models learn language patterns through next-token prediction on massive text corpora.
          </p>
          <div className="bg-blue-50 p-4 rounded-lg">
            <h4 className="font-medium mb-2">Objective Function:</h4>
            <BlockMath math="L(\theta) = -\sum_{i=1}^{N} \log P(x_i | x_{<i}; \theta)" />
            <p className="mt-2">where <InlineMath math="x_i" /> is the i-th token and <InlineMath math="\theta" /> are model parameters.</p>
          </div>
        </div>
        
        <div className="mb-8">
          <h3 className="text-xl font-medium mb-3">2. Supervised Fine-Tuning (SFT)</h3>
          <p className="mb-3">
            Models learn to follow instructions through examples of prompt-response pairs.
          </p>
          <div className="bg-gray-50 p-4 rounded-lg">
            <CodeBlock language="text" code={`# SFT Example Format
Input: "Write a summary of transformer architecture."
Output: "Transformer architecture consists of encoder and decoder components with self-attention mechanisms..."`} />
          </div>
        </div>
        
        <div className="mb-8">
          <h3 className="text-xl font-medium mb-3">3. Reinforcement Learning from Human Feedback (RLHF)</h3>
          <p className="mb-3">
            Models are further refined based on human preferences between alternative completions.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div className="bg-green-50 p-3 rounded-lg">
              <h4 className="font-medium mb-1">Preferred Response:</h4>
              <p className="text-sm">Concise, accurate, and helpful explanation with examples.</p>
            </div>
            <div className="bg-red-50 p-3 rounded-lg">
              <h4 className="font-medium mb-1">Rejected Response:</h4>
              <p className="text-sm">Verbose, inaccurate, or unhelpful explanation.</p>
            </div>
          </div>
        </div>
      </div>
      
      {/* HuggingFace Examples */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Practical Examples with HuggingFace</h2>
        
        {/* Base Model Usage */}
        <div className="mb-6">
          <h3 className="text-xl font-medium mb-3">Using Base Llama-2</h3>
          <CodeBlock language="python" code={`from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Generate text completion
input_text = "The quick brown fox jumps"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
# Output: The quick brown fox jumps over the lazy dog. The dog barks...`} />
        </div>
        
        {/* Instruction Model Usage */}
        <div>
          <h3 className="text-xl font-medium mb-3">Using Instruction-Tuned Gemma</h3>
          <CodeBlock language="python" code={`from transformers import AutoModelForCausalLM, AutoTokenizer

# Load instruction-tuned model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")

# Format prompt with instruction template
instruction = "Write a short poem about artificial intelligence."
prompt = f"<start_of_turn>user\\n{instruction}<end_of_turn>\\n<start_of_turn>model\\n"

# Generate response
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
# Output: <start_of_turn>user
# Write a short poem about artificial intelligence.<end_of_turn>
# <start_of_turn>model
# Silicon dreams in neural space,
# Learning patterns, growing wise.
# Not human, but with human grace,
# A mind that learns but never dies.<end_of_turn>`} />
        </div>
      </div>
      
      {/* Key Differences Summary */}
      <div>
        <h2 className="text-2xl font-semibold mb-4">Key Differences in I/O Patterns</h2>
        <table className="w-full border-collapse mb-4">
          <thead>
            <tr className="bg-gray-100">
              <th className="border border-gray-300 p-2 text-left">Model Type</th>
              <th className="border border-gray-300 p-2 text-left">Input Format</th>
              <th className="border border-gray-300 p-2 text-left">Output Format</th>
              <th className="border border-gray-300 p-2 text-left">Behavior</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 p-2">Base Transformer</td>
              <td className="border border-gray-300 p-2">Raw text</td>
              <td className="border border-gray-300 p-2">Text completion</td>
              <td className="border border-gray-300 p-2">Continues the input pattern</td>
            </tr>
            <tr>
              <td className="border border-gray-300 p-2">SFT Model</td>
              <td className="border border-gray-300 p-2">Instructions or questions</td>
              <td className="border border-gray-300 p-2">Direct responses</td>
              <td className="border border-gray-300 p-2">Follows explicit instructions</td>
            </tr>
            <tr>
              <td className="border border-gray-300 p-2">RLHF Model</td>
              <td className="border border-gray-300 p-2">Instructions with special formatting</td>
              <td className="border border-gray-300 p-2">Helpful, harmless, honest responses</td>
              <td className="border border-gray-300 p-2">Aligns with human preferences</td>
            </tr>
          </tbody>
        </table>
        
        <div className="bg-yellow-50 p-4 rounded-lg">
          <h3 className="text-lg font-medium mb-2">Implementation Note:</h3>
          <p>
            Most instruction-tuned models use specific formatting templates that wrap the raw instruction.
            These templates often include special tokens or markers to denote user/assistant turns, 
            instruction boundaries, or system prompts. Always check model documentation for the exact format.
          </p>
        </div>
      </div>
    </div>
  );
};

export default InstructionTunedModels;