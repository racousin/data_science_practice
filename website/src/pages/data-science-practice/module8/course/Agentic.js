import React from 'react';
import { Title, Text, List, Flex, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import DataInteractionPanel from 'components/DataInteractionPanel';

export default function Agentic() {
  // Notebook URLs for Career Coach Agent demonstration
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/course/module8_course_career_coach_agent.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/course/module8_course_career_coach_agent.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/course/module8_course_career_coach_agent.ipynb";

  const metadata = {
    description: "Complete Career Coach Agent: Build a multi-tool agentic system using LangGraph that analyzes CVs, researches markets, finds learning resources, searches jobs, generates cover letters, and provides negotiation guidance.",
    source: "LangGraph + OpenAI GPT-4",
    target: "Career coaching and job search optimization",
    listData: [
      { name: "Tools", description: "6 specialized tools (CV analysis, market research, job search, etc.)" },
      { name: "Architecture", description: "LangGraph state machine with conditional routing" },
      { name: "Demo", description: "Complete career analysis workflow with action tracking" }
    ],
  };

  return (
    <div>
      <div data-slide>
        <Title order={1} mb="lg">Agentic AI Systems</Title>

        <Text size="lg" mb="md">
          Traditional agents are processes written by experts to solve specific tasks by perceiving their environment, making decisions, and taking actions.
        </Text>

        <Text size="md" mb="md">
          The classical agent architecture follows a simple loop:
        </Text>

        <List spacing="sm" mb="lg" type="ordered">
          <List.Item>
            <Text weight={500}>Input:</Text> Perceive the current state of the environment
          </List.Item>
          <List.Item>
            <Text weight={500}>Decide:</Text> Use predefined rules or algorithms to choose an action
          </List.Item>
          <List.Item>
            <Text weight={500}>Act:</Text> Execute the chosen action and observe the result
          </List.Item>
        </List>

        <Text size="md" mb="md">
          Examples of traditional agents:
        </Text>

        <List spacing="xs" mb="md">
          <List.Item>Thermostat: Senses temperature, decides if heating/cooling needed, activates HVAC</List.Item>
          <List.Item>Chess engine: Analyzes board position, evaluates moves, selects optimal move</List.Item>
          <List.Item>Trading bot: Monitors market data, applies trading strategy, executes trades</List.Item>
        </List>

        <Text size="md" c="dimmed">
          These agents work well for static and well-defined problems where the rules and environment are known in advance. However, they struggle with ambiguity, natural language, and novel situations.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Limitations of Traditional Agents</Title>

        <Text size="md" mb="md">
          Traditional agents require explicit programming for every scenario. This becomes impractical for complex, open-ended tasks.
        </Text>

        <List spacing="md" mb="xl">
          <List.Item>
            <Text weight={500} mb="xs">Static decision logic</Text>
            <Text size="sm">Rules must be predefined by experts for every possible situation</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">No natural language understanding</Text>
            <Text size="sm">Cannot interpret ambiguous or context-dependent instructions</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">Limited adaptability</Text>
            <Text size="sm">Fail when encountering situations outside their programmed scenarios</Text>
          </List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# Traditional agent: rigid decision tree
def trading_agent(price, moving_avg):
    if price > moving_avg * 1.05:
        return "SELL"
    elif price < moving_avg * 0.95:
        return "BUY"
    else:
        return "HOLD"`}
        />

        <Text size="sm" mt="md" c="dimmed">
          Every decision rule must be explicitly coded. Adding new conditions requires modifying the code.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Limitations of Standalone LLMs</Title>

        <Text size="md" mb="md">
          While LLMs can understand natural language and generate human-like responses, they also have fundamental limitations when used alone:
        </Text>

        <List spacing="md" mb="xl">
          <List.Item>
            <Text weight={500} mb="xs">Fixed computational effort</Text>
            <Text size="sm">LLMs process simple questions ("2+2=?") with the same computational cost as complex problems, without the ability to allocate more reasoning for harder tasks</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">No tool access</Text>
            <Text size="sm">Cannot execute code, search the internet, or interact with external systems</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">Limited memory</Text>
            <Text size="sm">Context window constraints prevent access to large knowledge bases (though RAG helps address this)</Text>
          </List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# LLM alone cannot execute this
response = llm.generate("What's the weather in Paris?")
# LLM can only generate text based on training data
# It cannot actually fetch current weather data`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="md">The Agentic Approach</Title>

        <Text size="md" mb="md">
          Agentic AI combines the flexibility of LLMs with the action-taking capabilities of traditional agents. The key insight: replace the rigid decision component with an LLM.
        </Text>

        <Text size="md" mb="md">
          The architecture remains the same Input → Decide → Act loop, but the decision-making is now powered by natural language reasoning:
        </Text>

        <List spacing="sm" mb="lg" type="ordered">
          <List.Item>
            <Text weight={500}>Input:</Text> Agent receives a task in natural language
          </List.Item>
          <List.Item>
            <Text weight={500}>Decide (LLM):</Text> LLM reasons about which tool or action to use based on a prompt
          </List.Item>
          <List.Item>
            <Text weight={500}>Act:</Text> Agent executes the chosen tool and observes results
          </List.Item>
        </List>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/agentic.jpg"
            alt="Agentic architecture: traditional agent loop with LLM-powered decision making"
            style={{ maxWidth: "70%", height: "auto" }}
          />
        </Flex>

        <Text size="md" c="dimmed">
          By prompting the LLM to choose from defined tools and other agents, we create flexible systems that can handle ambiguous instructions and novel situations while still taking concrete actions.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Reflection Agents</Title>

        <Text size="md" mb="md">
          A reflection agent improves its outputs by critiquing and refining its own work through iterative self-evaluation.
        </Text>

        <Text size="md" mb="md">
          The reflection loop:
        </Text>

        <List spacing="xs" mb="lg" type="ordered">
          <List.Item>Generate initial response to the task</List.Item>
          <List.Item>Critique the response (identify weaknesses, errors, improvements)</List.Item>
          <List.Item>Generate improved version based on critique</List.Item>
          <List.Item>Repeat until quality threshold is met</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# Reflection pattern
def reflection_agent(task):
    response = llm.generate(task)
    for iteration in range(max_iterations):
        critique = llm.generate(
            f"Critique this response: {response}"
        )
        if "acceptable" in critique.lower():
            break
        response = llm.generate(
            f"Improve based on: {critique}"
        )
    return response`}
        />

        <Text size="md" mt="md">
          This pattern is effective for tasks requiring high quality outputs such as code generation, writing, and problem-solving where iteration leads to improvement.
        </Text>
                <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/critic.png"
            alt="Agentic architecture: traditional agent loop with LLM-powered decision making"
            style={{ maxWidth: "70%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Agent Tools</Title>

        <Text size="md" mb="md">
          Tools extend agents beyond text generation, enabling interaction with external systems. Each tool is a function the agent can invoke based on its reasoning.
        </Text>

        <Title order={3} size="h4" mb="sm">Common Tool Categories</Title>

        <List spacing="sm" mb="lg">
          <List.Item>
            <Text weight={500}>Internet access:</Text> Search engines, web scraping, API calls to external services
          </List.Item>
          <List.Item>
            <Text weight={500}>Code execution:</Text> Python interpreter, shell commands, computational operations
          </List.Item>
          <List.Item>
            <Text weight={500}>Model Context Protocol (MCP):</Text> Standardized way to connect LLMs to data sources and tools
          </List.Item>
          <List.Item>
            <Text weight={500}>Vector databases (RAG):</Text> Semantic search over document collections for knowledge retrieval
          </List.Item>
        </List>

        <Text size="md" mb="sm">
          Tools are defined with a name, description, and function signature. The LLM uses the description to decide when to invoke each tool.
        </Text>

        <CodeBlock
          language="python"
          code={`tools = [
    {
        "name": "web_search",
        "description": "Search the internet for current information",
        "function": lambda query: search_api(query)
    },
    {
        "name": "python_exec",
        "description": "Execute Python code for calculations",
        "function": lambda code: exec_code(code)
    }
]`}
        />
      </div>

      <div data-slide>
        <Title order={2} mb="md">Tool Usage Example with LangGraph</Title>

        <Text size="md" mb="md">
          LangGraph provides a framework for building agents with tool-calling capabilities. Here's a simple example:
        </Text>

        <CodeBlock
          language="python"
          code={`from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information on the internet."""
    return f"Search results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))`}
        />

        <Text size="md" mt="md" mb="sm">
          Create and run the agent:
        </Text>

        <CodeBlock
          language="python"
          code={`llm = ChatOpenAI(model="gpt-4")
tools = [search, calculator]

agent = create_react_agent(llm, tools)

# Agent decides which tools to use
response = agent.invoke({
    "messages": [("user", "What is 25 * 17?")]
})

# Agent will call calculator("25 * 17") and return 425`}
        />
                        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/reflexion.png"
            alt="Agentic architecture: traditional agent loop with LLM-powered decision making"
            style={{ maxWidth: "70%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Advanced Agentic Architectures</Title>

        <Text size="md" mb="md">
          Beyond simple tool-calling agents, more sophisticated architectures enable complex reasoning and task decomposition:
        </Text>

        <List spacing="sm" mb="lg">
          <List.Item>
            <Text weight={500}>Planner agents:</Text> Decompose complex tasks into sequential steps before execution
          </List.Item>
          <List.Item>
            <Text weight={500}>Tree-of-Thought (ToT):</Text> Explore multiple reasoning paths simultaneously and select the best approach
          </List.Item>
          <List.Item>
            <Text weight={500}>Multi-agent systems:</Text> Multiple specialized agents collaborate, each with specific expertise and tools
          </List.Item>
        </List>

                <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/agenticarch.gif"
            alt="Agentic architecture: traditional agent loop with LLM-powered decision making"
            style={{ maxWidth: "70%", height: "auto" }}
          />
        </Flex>

        <Text size="md" c="dimmed">
          These architectures build on the fundamental agentic pattern, adding layers of planning, exploration, and collaboration to handle increasingly complex tasks.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mb="md">Maximizing LLM Value</Title>

        <Text size="md" mb="md">
          The agentic framework represents an accessible approach to unlock the full potential of LLMs by combining their strengths with practical capabilities:
        </Text>

        <List spacing="md" mb="lg">
          <List.Item>
            <Text weight={500} mb="xs">Natural language interface</Text>
            <Text size="sm">Users can specify complex tasks in plain language rather than code</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">Dynamic tool selection</Text>
            <Text size="sm">LLM intelligently chooses appropriate tools based on task requirements</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">Iterative refinement</Text>
            <Text size="sm">Agents can reflect on results and adjust their approach</Text>
          </List.Item>

          <List.Item>
            <Text weight={500} mb="xs">Extensible capabilities</Text>
            <Text size="sm">Adding new tools or agents extends functionality without rewriting core logic</Text>
          </List.Item>
        </List>

        <Text size="md">
          By structuring LLM interactions as agentic systems with tools, memory, and reflection, we transform static language models into dynamic problem-solving systems capable of handling real-world tasks autonomously.
        </Text>
      </div>

      <DataInteractionPanel
        notebookUrl={notebookUrl}
        notebookHtmlUrl={notebookHtmlUrl}
        notebookColabUrl={notebookColabUrl}
        metadata={metadata}
      />

    </div>
  );
}
