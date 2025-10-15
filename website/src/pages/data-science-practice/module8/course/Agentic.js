import React from 'react';
import { Text, Title, List, Table, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

export default function Agentic() {
  return (
    <div>
      <div data-slide>
        <Title order={1}>Agentic AI Systems</Title>
        <Text size="xl" mt="md">
          Building autonomous AI agents that perceive, reason, and act
        </Text>
        <Text mt="lg">
          This module explores the architecture and implementation of AI agents that can autonomously
          make decisions, use tools, maintain memory, and execute complex multi-step tasks.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>What are AI Agents?</Title>
        <Text mt="md">
          An AI agent is an autonomous system that perceives its environment, reasons about it,
          and takes actions to achieve specific goals. Unlike simple query-response systems,
          agents maintain state, plan sequences of actions, and adapt based on feedback.
        </Text>
        <Text mt="md">
          Key characteristics of AI agents:
        </Text>
        <List mt="sm" spacing="sm">
          <List.Item><strong>Autonomy</strong>: Operate without constant human intervention</List.Item>
          <List.Item><strong>Reactivity</strong>: Perceive and respond to environmental changes</List.Item>
          <List.Item><strong>Pro-activeness</strong>: Take initiative to achieve goals</List.Item>
          <List.Item><strong>Social ability</strong>: Interact with other agents or humans</List.Item>
        </List>
        <Text mt="md">
          Modern LLM-based agents combine language understanding with tool use and reasoning
          to accomplish complex tasks across multiple steps.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Agent Architecture Components</Title>
        <Text mt="md">
          An AI agent consists of three fundamental components that form a perception-action cycle:
        </Text>
        <Table mt="md" withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Component</Table.Th>
              <Table.Th>Function</Table.Th>
              <Table.Th>Examples</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td><strong>Perception</strong></Table.Td>
              <Table.Td>Gather information from environment</Table.Td>
              <Table.Td>API calls, database queries, sensor data</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><strong>Reasoning</strong></Table.Td>
              <Table.Td>Process information and make decisions</Table.Td>
              <Table.Td>LLM inference, planning, tool selection</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><strong>Action</strong></Table.Td>
              <Table.Td>Execute decisions in environment</Table.Td>
              <Table.Td>Tool execution, API writes, file operations</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
        <Text mt="md">
          The agent loops through these components: perceiving the current state, reasoning about
          the best action, executing that action, and perceiving the results.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Types of AI Agents</Title>
        <Text mt="md">
          AI agents can be classified by their decision-making approach:
        </Text>
        <Title order={4} mt="lg">Reactive Agents</Title>
        <Text mt="sm">
          Respond directly to current perceptions without internal state. Fast but limited.
        </Text>
        <CodeBlock
          language="python"
          code={`def reactive_agent(perception):
    return action_mapping[perception]  # Direct mapping`}
        />
        <Title order={4} mt="lg">Deliberative Agents</Title>
        <Text mt="sm">
          Maintain internal world model, plan ahead, and reason about consequences.
        </Text>
        <CodeBlock
          language="python"
          code={`def deliberative_agent(perception, world_model):
    world_model.update(perception)
    plan = planner.create_plan(world_model, goal)
    return plan.next_action()`}
        />
        <Title order={4} mt="lg">Hybrid Agents</Title>
        <Text mt="sm">
          Combine reactive responses for urgent situations with deliberative planning for complex goals.
          Most modern LLM agents are hybrid, reacting to immediate context while maintaining longer-term plans.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Agent Framework: ReAct</Title>
        <Text mt="md">
          ReAct (Reasoning + Acting) is a paradigm where agents interleave reasoning traces
          with action execution. Instead of just predicting actions, the agent explicitly
          generates reasoning steps.
        </Text>
        <Text mt="md">
          The ReAct pattern:
        </Text>
        <List mt="sm" spacing="sm">
          <List.Item><strong>Thought</strong>: Agent reasons about current state and next step</List.Item>
          <List.Item><strong>Action</strong>: Agent calls a tool or takes an action</List.Item>
          <List.Item><strong>Observation</strong>: Agent receives feedback from action</List.Item>
          <List.Item>Loop continues until goal is achieved</List.Item>
        </List>
        <Text mt="md">
          This interleaving allows agents to adjust their reasoning based on actual outcomes,
          making them more robust to errors and unexpected situations.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>ReAct Mathematical Formulation</Title>
        <Text mt="md">
          The agent's decision at timestep <InlineMath>t</InlineMath> is formalized as:
        </Text>
        <BlockMath>{`a_t = \\pi(s_t, m_t, h_t)`}</BlockMath>
        <Text mt="sm">
          where:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath>a_t</InlineMath>: action at time t</List.Item>
          <List.Item><InlineMath>\pi</InlineMath>: agent policy (typically an LLM)</List.Item>
          <List.Item><InlineMath>s_t</InlineMath>: current state/observation</List.Item>
          <List.Item><InlineMath>m_t</InlineMath>: memory/context</List.Item>
          <List.Item><InlineMath>h_t</InlineMath>: history of thoughts and actions</List.Item>
        </List>
        <Text mt="md">
          The ReAct trajectory is a sequence:
        </Text>
        <BlockMath>{`\\tau = (t_1, a_1, o_1, t_2, a_2, o_2, ..., t_n, a_n, o_n)`}</BlockMath>
        <Text mt="sm">
          where <InlineMath>t_i</InlineMath> are thoughts (reasoning traces), <InlineMath>a_i</InlineMath> are
          actions, and <InlineMath>o_i</InlineMath> are observations from the environment.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>ReAct Process Example</Title>
        <Text mt="md">
          Here's how ReAct works for a question answering task:
        </Text>
        <Text mt="md" style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', background: '#f5f5f5', padding: '12px' }}>
{`Question: What is the population of the capital of France?

Thought 1: I need to first identify the capital of France.
Action 1: Search[capital of France]
Observation 1: The capital of France is Paris.

Thought 2: Now I need to find the population of Paris.
Action 2: Search[population of Paris]
Observation 2: Paris has approximately 2.2 million inhabitants.

Thought 3: I have the answer.
Action 3: Finish[2.2 million]`}
        </Text>
        <Text mt="md">
          The agent alternates between thinking (reasoning about what to do), acting (using tools),
          and observing (receiving results), building toward the final answer iteratively.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Agent Tools and Actions</Title>
        <Text mt="md">
          Tools extend an agent's capabilities beyond text generation. Each tool is a function
          the agent can invoke to interact with external systems.
        </Text>
        <Text mt="md">
          Common tool categories:
        </Text>
        <List mt="sm" spacing="sm">
          <List.Item><strong>Search</strong>: Web search, database queries, document retrieval</List.Item>
          <List.Item><strong>Computation</strong>: Calculator, code execution, mathematical operations</List.Item>
          <List.Item><strong>Data access</strong>: API calls, file reading, database access</List.Item>
          <List.Item><strong>Data modification</strong>: File writing, API posts, database updates</List.Item>
          <List.Item><strong>Communication</strong>: Email, messaging, notifications</List.Item>
        </List>
        <Text mt="md">
          The agent's policy must learn to select the appropriate tool for each step:
        </Text>
        <BlockMath>{`\\text{tool}^* = \\arg\\max_{\\text{tool}} P(\\text{tool} | q, c, h)`}</BlockMath>
        <Text mt="sm">
          where <InlineMath>q</InlineMath> is the query, <InlineMath>c</InlineMath> is context,
          and <InlineMath>h</InlineMath> is history.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Tool Definition and Registration</Title>
        <Text mt="md">
          Tools are defined with name, description, and parameter schemas. The description
          is critical as the LLM uses it to decide when to use the tool.
        </Text>
        <CodeBlock
          language="python"
          code={`from langchain.tools import Tool

def calculator(expression: str) -> str:
    return str(eval(expression))`}
        />
        <Text mt="sm">
          Register the tool with metadata:
        </Text>
        <CodeBlock
          language="python"
          code={`calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Useful for mathematical calculations"
)`}
        />
        <Text mt="sm">
          The agent receives tool descriptions in its prompt and learns to call them appropriately
          based on the task requirements.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Agent Memory Systems</Title>
        <Text mt="md">
          Memory allows agents to maintain context across interactions and learn from experience.
        </Text>
        <Title order={4} mt="lg">Short-term Memory</Title>
        <Text mt="sm">
          Conversation history within current session. Stored in the prompt context window.
        </Text>
        <Title order={4} mt="lg">Long-term Memory</Title>
        <Text mt="sm">
          Persistent storage across sessions. Typically uses vector databases for semantic retrieval.
        </Text>
        <BlockMath>{`m_t = \\text{retrieve}(\\text{VectorDB}, s_t, k=5)`}</BlockMath>
        <Title order={4} mt="lg">Episodic Memory</Title>
        <Text mt="sm">
          Records of past experiences and their outcomes. Used for learning and adaptation.
        </Text>
        <Text mt="md">
          Memory management is crucial for token efficiency. Agents must balance maintaining
          sufficient context with staying within model token limits.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Planning Strategies</Title>
        <Text mt="md">
          Agents use various strategies to plan multi-step tasks:
        </Text>
        <Title order={4} mt="lg">Chain-of-Thought (CoT)</Title>
        <Text mt="sm">
          Generate intermediate reasoning steps sequentially. Simple but effective.
        </Text>
        <CodeBlock
          language="text"
          code={`Step 1: Identify the problem
Step 2: Break down into sub-tasks
Step 3: Solve each sub-task`}
        />
        <Title order={4} mt="lg">Tree-of-Thought (ToT)</Title>
        <Text mt="sm">
          Explore multiple reasoning paths, evaluate them, and select the best. More robust but costly.
        </Text>
        <Title order={4} mt="lg">Plan-and-Execute</Title>
        <Text mt="sm">
          First create a complete plan, then execute each step. Useful for complex multi-step tasks.
        </Text>
        <CodeBlock
          language="python"
          code={`plan = planner.create_plan(task)
for step in plan:
    result = executor.execute(step)
    plan.update(result)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>LangChain Agents Overview</Title>
        <Text mt="md">
          LangChain provides a framework for building agents with standardized components:
        </Text>
        <List mt="md" spacing="sm">
          <List.Item><strong>Agent</strong>: Core decision-making component using an LLM</List.Item>
          <List.Item><strong>Tools</strong>: Functions the agent can call</List.Item>
          <List.Item><strong>Toolkit</strong>: Pre-built collections of related tools</List.Item>
          <List.Item><strong>AgentExecutor</strong>: Runtime that manages the agent loop</List.Item>
          <List.Item><strong>Memory</strong>: Conversation history and context management</List.Item>
        </List>
        <Text mt="md">
          The AgentExecutor handles the ReAct loop: prompting the LLM, parsing tool calls,
          executing tools, and feeding observations back to the LLM until the task completes.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Agent Types in LangChain</Title>
        <Text mt="md">
          LangChain supports multiple agent architectures:
        </Text>
        <Title order={4} mt="lg">Zero-shot ReAct</Title>
        <Text mt="sm">
          Uses tool descriptions to decide which tools to use without examples. Works with any tools.
        </Text>
        <Title order={4} mt="lg">Conversational Agent</Title>
        <Text mt="sm">
          Optimized for multi-turn conversations with memory. Maintains dialogue context.
        </Text>
        <CodeBlock
          language="python"
          code={`from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()`}
        />
        <Title order={4} mt="lg">OpenAI Functions Agent</Title>
        <Text mt="sm">
          Uses OpenAI's function calling API for structured tool invocation. More reliable parsing.
        </Text>
        <CodeBlock
          language="python"
          code={`agent = create_openai_functions_agent(
    llm, tools, prompt
)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Creating a Simple Agent</Title>
        <Text mt="md">
          Basic agent setup with LangChain:
        </Text>
        <CodeBlock
          language="python"
          code={`from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool`}
        />
        <Text mt="sm">
          Define the LLM and tools:
        </Text>
        <CodeBlock
          language="python"
          code={`llm = ChatOpenAI(temperature=0, model="gpt-4")

tools = [
    Tool(name="Search", func=search_func,
         description="Search for information"),
    Tool(name="Calculator", func=calc_func,
         description="Perform calculations")
]`}
        />
        <Text mt="sm">
          Create and run the agent:
        </Text>
        <CodeBlock
          language="python"
          code={`agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "What is 25 * 17?"})`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Custom Tool Creation</Title>
        <Text mt="md">
          Create custom tools with proper type hints and descriptions:
        </Text>
        <CodeBlock
          language="python"
          code={`from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: City name or zip code
    """
    # API call to weather service
    return f"Weather in {location}: Sunny, 72°F"`}
        />
        <Text mt="sm">
          Tools with structured inputs:
        </Text>
        <CodeBlock
          language="python"
          code={`from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    location: str = Field(description="City name")
    units: str = Field(description="celsius or fahrenheit")

@tool(args_schema=WeatherInput)
def get_weather_detailed(location: str, units: str) -> str:
    """Get weather with specific units."""
    return f"Temp: 22°C" if units == "celsius" else "Temp: 72°F"`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Agent Execution Loop</Title>
        <Text mt="md">
          The agent executor runs a loop until completion or max iterations:
        </Text>
        <CodeBlock
          language="python"
          code={`while not done and iterations < max_iterations:
    # 1. LLM decides next action
    action = agent.plan(state, memory)

    # 2. Execute tool if action specified
    if action.tool:
        observation = tools[action.tool].run(action.input)

    # 3. Update state and memory
    state.update(observation)
    memory.add(action, observation)

    # 4. Check if task complete
    done = action.is_final()`}
        />
        <Text mt="md">
          The executor handles errors, retries, and token management. It also provides callbacks
          for logging and monitoring agent behavior during execution.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Multi-Agent Systems</Title>
        <Text mt="md">
          Multiple agents can collaborate to solve complex tasks by dividing work and specializing:
        </Text>
        <List mt="md" spacing="sm">
          <List.Item><strong>Hierarchical</strong>: Manager agent delegates to worker agents</List.Item>
          <List.Item><strong>Collaborative</strong>: Peer agents work together on shared goals</List.Item>
          <List.Item><strong>Competitive</strong>: Agents propose solutions, best one selected</List.Item>
          <List.Item><strong>Sequential</strong>: Output of one agent feeds into next</List.Item>
        </List>
        <Text mt="md">
          Multi-agent systems can achieve better performance through specialization:
        </Text>
        <CodeBlock
          language="python"
          code={`researcher = Agent(role="Researcher", tools=[search, read])
writer = Agent(role="Writer", tools=[write, edit])
reviewer = Agent(role="Reviewer", tools=[review])`}
        />
        <Text mt="sm">
          Each agent focuses on its expertise, and a coordinator manages the workflow.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Agent Communication Protocols</Title>
        <Text mt="md">
          Agents communicate through structured message passing:
        </Text>
        <CodeBlock
          language="python"
          code={`class Message:
    sender: str
    receiver: str
    content: dict
    message_type: str  # request, response, broadcast`}
        />
        <Text mt="sm">
          Example communication flow:
        </Text>
        <CodeBlock
          language="python"
          code={`# Manager requests task from worker
manager.send(Message(
    sender="manager",
    receiver="researcher",
    content={"task": "Find data on topic X"},
    message_type="request"
))`}
        />
        <Text mt="md">
          Communication protocols define how agents coordinate, share information, and resolve conflicts.
          Common patterns include request-response, publish-subscribe, and blackboard systems.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>AutoGPT Architecture</Title>
        <Text mt="md">
          AutoGPT is an autonomous agent that breaks down goals into tasks and executes them
          with minimal human intervention.
        </Text>
        <Text mt="md">
          Key components:
        </Text>
        <List mt="sm" spacing="sm">
          <List.Item><strong>Goal management</strong>: Maintains high-level objectives</List.Item>
          <List.Item><strong>Task generation</strong>: Breaks goals into actionable tasks</List.Item>
          <List.Item><strong>Memory system</strong>: Vector database for context</List.Item>
          <List.Item><strong>Self-criticism</strong>: Evaluates its own outputs</List.Item>
          <List.Item><strong>Internet access</strong>: Web search and browsing</List.Item>
        </List>
        <Text mt="md">
          AutoGPT loop:
        </Text>
        <CodeBlock
          language="python"
          code={`while goal_not_achieved:
    tasks = generate_tasks(goal, context)
    for task in prioritize_tasks(tasks):
        result = execute_task(task)
        memory.store(task, result)
        self_evaluate(result)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>BabyAGI Architecture</Title>
        <Text mt="md">
          BabyAGI is a simplified autonomous agent focused on task management and execution.
        </Text>
        <Text mt="md">
          Core loop with three components:
        </Text>
        <List mt="sm" spacing="sm">
          <List.Item><strong>Execution agent</strong>: Completes current task using context</List.Item>
          <List.Item><strong>Task creation agent</strong>: Generates new tasks based on results</List.Item>
          <List.Item><strong>Prioritization agent</strong>: Reorders task list by importance</List.Item>
        </List>
        <CodeBlock
          language="python"
          code={`task_list = [initial_task]

while task_list:
    current_task = task_list.pop(0)
    result = execution_agent(current_task)
    new_tasks = task_creation_agent(result)
    task_list.extend(new_tasks)
    task_list = prioritization_agent(task_list)`}
        />
        <Text mt="md">
          BabyAGI uses vector embeddings to maintain context and prioritize tasks effectively.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Agent Evaluation Metrics</Title>
        <Text mt="md">
          Evaluating agent performance requires multiple dimensions:
        </Text>
        <Title order={4} mt="lg">Task Success Rate</Title>
        <Text mt="sm">
          Percentage of tasks completed successfully within constraints.
        </Text>
        <BlockMath>{`\\text{Success Rate} = \\frac{\\text{Successful Tasks}}{\\text{Total Tasks}}`}</BlockMath>
        <Title order={4} mt="lg">Efficiency Metrics</Title>
        <List mt="sm" spacing="xs">
          <List.Item>Number of steps/actions taken</List.Item>
          <List.Item>Token usage and API costs</List.Item>
          <List.Item>Execution time</List.Item>
        </List>
        <Title order={4} mt="lg">Quality Metrics</Title>
        <List mt="sm" spacing="xs">
          <List.Item>Correctness of final output</List.Item>
          <List.Item>Appropriateness of tool selection</List.Item>
          <List.Item>Reasoning coherence</List.Item>
        </List>
        <Title order={4} mt="lg">Robustness</Title>
        <Text mt="sm">
          Performance under errors, ambiguous instructions, and edge cases.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Agent Failure Modes and Safety</Title>
        <Text mt="md">
          Common failure modes in agent systems:
        </Text>
        <List mt="md" spacing="sm">
          <List.Item><strong>Infinite loops</strong>: Agent repeats same action without progress</List.Item>
          <List.Item><strong>Hallucinated tools</strong>: Attempts to use non-existent tools</List.Item>
          <List.Item><strong>Context drift</strong>: Loses track of original goal</List.Item>
          <List.Item><strong>Tool misuse</strong>: Uses tools incorrectly or unsafely</List.Item>
          <List.Item><strong>Resource exhaustion</strong>: Exceeds token limits or API quotas</List.Item>
        </List>
        <Text mt="md">
          Safety mechanisms:
        </Text>
        <CodeBlock
          language="python"
          code={`agent = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=15,        # Prevent infinite loops
    max_execution_time=300,   # Time limit
    handle_parsing_errors=True # Graceful error handling
)`}
        />
        <Text mt="sm">
          Additional safety: human-in-the-loop for critical actions, tool sandboxing, and output validation.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Practical Agent Implementation</Title>
        <Text mt="md">
          Complete example of a research assistant agent:
        </Text>
        <CodeBlock
          language="python"
          code={`from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder`}
        />
        <Text mt="sm">
          Define specialized tools:
        </Text>
        <CodeBlock
          language="python"
          code={`@tool
def search_papers(query: str) -> str:
    """Search academic papers on a topic."""
    # Call to paper database API
    return "Found 5 papers on " + query

@tool
def summarize_paper(paper_id: str) -> str:
    """Get summary of a specific paper."""
    # Retrieve and summarize paper
    return "Paper summary: ..."`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Practical Agent Implementation (continued)</Title>
        <Text mt="md">
          Create the agent with custom prompt:
        </Text>
        <CodeBlock
          language="python"
          code={`prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Help users find and understand academic papers."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [search_papers, summarize_paper]

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)`}
        />
        <Text mt="sm">
          Execute research task:
        </Text>
        <CodeBlock
          language="python"
          code={`result = executor.invoke({
    "input": "Find papers on transformers in NLP and summarize the key findings"
})`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Advanced Agent Patterns</Title>
        <Text mt="md">
          Sophisticated agent architectures for complex scenarios:
        </Text>
        <Title order={4} mt="lg">Hierarchical Agents</Title>
        <Text mt="sm">
          Manager agent decomposes tasks and delegates to specialized worker agents.
        </Text>
        <CodeBlock
          language="python"
          code={`class ManagerAgent:
    def execute(self, task):
        subtasks = self.decompose(task)
        results = [worker.execute(st) for worker, st in
                   zip(self.workers, subtasks)]
        return self.synthesize(results)`}
        />
        <Title order={4} mt="lg">Collaborative Agents</Title>
        <Text mt="sm">
          Multiple agents work together, sharing a common workspace and communicating peer-to-peer.
        </Text>
        <CodeBlock
          language="python"
          code={`workspace = SharedWorkspace()
agents = [ResearchAgent(), WriterAgent(), EditorAgent()]
for agent in agents:
    agent.contribute(workspace)
final_output = workspace.get_result()`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Agent Orchestration</Title>
        <Text mt="md">
          Orchestration manages agent coordination, resource allocation, and workflow:
        </Text>
        <CodeBlock
          language="python"
          code={`class AgentOrchestrator:
    def __init__(self, agents, workflow):
        self.agents = agents
        self.workflow = workflow

    def execute(self, task):
        state = {"task": task}
        for step in self.workflow:
            agent = self.agents[step.agent_name]
            result = agent.execute(state)
            state.update(result)
        return state["output"]`}
        />
        <Text mt="md">
          Orchestrators can implement complex patterns:
        </Text>
        <List mt="sm" spacing="xs">
          <List.Item>Conditional routing based on intermediate results</List.Item>
          <List.Item>Parallel execution of independent sub-tasks</List.Item>
          <List.Item>Error recovery and retry logic</List.Item>
          <List.Item>Dynamic agent selection based on task requirements</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>ReAct Agent Flow Diagram</Title>
        <Flex justify="center" mt="md">
          <Image
            src="/api/placeholder/800/500"
            alt="ReAct agent decision-making flow showing the cycle of Thought, Action, and Observation"
            style={{ maxWidth: '100%' }}
          />
        </Flex>
        <Text mt="md" size="sm" style={{ textAlign: 'center' }}>
          The ReAct cycle: Agent generates thoughts (reasoning), executes actions (tool calls),
          receives observations (results), and repeats until task completion.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Multi-Agent System Architecture</Title>
        <Flex justify="center" mt="md">
          <Image
            src="/api/placeholder/800/500"
            alt="Multi-agent system showing hierarchical coordination with manager and specialized worker agents"
            style={{ maxWidth: '100%' }}
          />
        </Flex>
        <Text mt="md" size="sm" style={{ textAlign: 'center' }}>
          Hierarchical multi-agent architecture: Manager agent coordinates specialized worker agents
          (researcher, analyzer, writer) with shared memory and communication protocols.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Agent Decision-Making Process</Title>
        <Flex justify="center" mt="md">
          <Image
            src="/api/placeholder/800/500"
            alt="Agent decision-making process showing perception, reasoning, tool selection, and action execution"
            style={{ maxWidth: '100%' }}
          />
        </Flex>
        <Text mt="md" size="sm" style={{ textAlign: 'center' }}>
          Agent decision pipeline: Input processing, memory retrieval, reasoning/planning,
          tool selection, action execution, and observation feedback loop.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>References</Title>
        <Text mt="md" weight={500}>Key Papers and Resources:</Text>
        <List mt="md" spacing="md">
          <List.Item>
            <strong>ReAct: Synergizing Reasoning and Acting in Language Models</strong>
            <br />
            Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022)
            <br />
            Introduces the ReAct paradigm for interleaving reasoning and action in LLM agents
          </List.Item>
          <List.Item>
            <strong>AutoGPT</strong>
            <br />
            Autonomous agent framework: github.com/Significant-Gravitas/AutoGPT
            <br />
            Open-source autonomous agent with goal decomposition and self-evaluation
          </List.Item>
          <List.Item>
            <strong>BabyAGI</strong>
            <br />
            Task-driven autonomous agent: github.com/yoheinakajima/babyagi
            <br />
            Minimalist agent focusing on task creation, prioritization, and execution
          </List.Item>
          <List.Item>
            <strong>LangChain Documentation</strong>
            <br />
            python.langchain.com/docs/modules/agents/
            <br />
            Comprehensive guide to building agents with LangChain framework
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>Summary</Title>
        <Text mt="md">
          AI agents represent a paradigm shift from single-turn interactions to autonomous,
          multi-step task execution. Key takeaways:
        </Text>
        <List mt="md" spacing="sm">
          <List.Item>Agents combine perception, reasoning, and action in iterative loops</List.Item>
          <List.Item>ReAct pattern enables robust decision-making through explicit reasoning</List.Item>
          <List.Item>Tools extend agent capabilities to interact with external systems</List.Item>
          <List.Item>Memory systems allow agents to maintain context and learn from experience</List.Item>
          <List.Item>Multi-agent systems enable specialization and collaboration</List.Item>
          <List.Item>LangChain provides practical framework for agent implementation</List.Item>
          <List.Item>Safety mechanisms are essential for reliable agent deployment</List.Item>
        </List>
        <Text mt="lg">
          The agent paradigm is rapidly evolving, with improvements in planning, tool use,
          error recovery, and multi-agent coordination. As LLMs become more capable,
          agents will handle increasingly complex and open-ended tasks.
        </Text>
      </div>
    </div>
  );
}
