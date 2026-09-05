# Agents

An agent is a language model in a loop with tools. That is the whole idea. The
engineering is entirely in the loop's guard rails, because the model will
happily run forever and report success.

<!-- notes: 30 minutes. They already supervise a coding agent from the 12h
module — start by asking what it does badly and map those answers onto the
failure slide. Do not teach LangGraph; teach the contract. -->

---

## A model in a loop

![An LLM replacing the decision step of a classical agent loop](/api/academic_courses/assets/lessons/108/agentic.jpg)

The classical agent is perceive, decide, act — with the decision hand-coded.
A trading rule, a thermostat, a chess engine. Rigid, and unable to handle
anything its author did not anticipate.

Replace the decision step with an LLM and the loop takes natural-language goals
and picks from a set of tools: the model contributes judgement, the tools the
ability to actually do something.

```python
while not done:
    action = llm(goal, history, tools)   # decide
    result = execute(action)             # act
    history.append((action, result))     # perceive
```

---

## The tool-calling contract

A tool is a JSON schema in, a structured call out, a result back. The model
never runs anything — it emits a request that your code validates and executes.

```python
@tool
def search_orders(customer_id: str, since: str) -> list[dict]:
    """Return orders for a customer placed on or after an ISO date."""
```

The docstring and the type hints *are* the interface: they become the schema
the model reads when deciding. A vague description is a tool that gets called
at the wrong time, and it is a prompt engineering bug, not a code bug.

Validate arguments at the boundary: the model produces text, so treat every tool
call as untrusted input and let a Pydantic model raise on anything malformed.

**MCP** — the Model Context Protocol — is the standard for publishing this
contract. A server declares its tools once and any compatible client can call
them, which is how your coding agent reaches a database or a ticket tracker. It
solves distribution, not judgement: a badly described tool stays badly described
over MCP.

---

## ReAct

Interleave reasoning and acting rather than planning everything first:

```text
Thought: I need the order history before I can answer.
Action: search_orders(customer_id="c-91", since="2026-01-01")
Observation: [ ... 3 orders ... ]
Thought: The last order was refunded. That answers the question.
```

Each observation conditions the next thought, so the agent recovers from a tool
returning something unexpected. This is the default loop, and what every
tool-calling API implements underneath.

Its weakness is myopia: it optimises the next step, not the trajectory, and on a
ten-step task will walk down a branch that cannot succeed.

---

## Planning and decomposition

For multi-step work, generate a plan first, then execute the steps as separate
calls with their own context.

- **Plan-and-execute** — one planning call, N execution calls. Cheaper, and the
  plan is inspectable before anything runs.
- **Re-planning** — revise when a step fails, rather than retrying it forever.
- **Decomposition** — a subtask with a narrow tool set and a short context beats
  one agent holding everything in mind.

A written plan is also the artefact a human can approve. For anything that
writes, spends or sends, the plan is the approval point.

---

## Memory

| Kind | Lives in | Holds |
|---|---|---|
| Working | the context window | this task's messages and observations |
| Episodic | a store keyed by session | what happened in earlier runs |
| Semantic | a vector index | facts and documents (this is RAG) |

Working memory is the one that breaks. Observations accumulate, the context
fills, and cost grows with the square of the step count if you resend
everything.

Summarise old turns, drop tool outputs once used, and cap transcript tokens. An
agent whose context is 90% stale tool output decides worse than one with a
short, curated history.

---

## Reflection

![The generate–critique–revise loop](/api/academic_courses/assets/lessons/108/reflexion.png)

Generate, criticise the output against a rubric, revise. Repeat until the
critique is clean or a budget runs out.

```python
for _ in range(MAX_REVISIONS):
    critique = llm(CRITIQUE_PROMPT.format(task=task, draft=draft))
    if critique["ok"]:
        break
    draft = llm(REVISE_PROMPT.format(draft=draft, critique=critique))
```

It works when the critique has ground truth to lean on — a failing test, a type
error, a schema violation. It degrades into self-congratulation when the only
judge is the same model that wrote the text. Ground the critic in something
external, or do not run the loop.

---

## Multi-agent patterns

![Multi-agent architectures: supervisor, pipeline, debate](/api/academic_courses/assets/lessons/108/agenticarch.gif)

- **Supervisor** — a router delegates to specialists and assembles the result.
- **Pipeline** — fixed hand-off, each stage with its own tools.
- **Debate** — several agents argue; a judge picks. Expensive, occasionally
  worth it on reasoning tasks.

Every extra agent adds a serialisation boundary, a full context copy and a place
for the task description to drift. Most problems solved by "add another agent"
are solved more cheaply by better tools or a shorter context.

Reach for multi-agent when the subtasks genuinely need different tools or
different permissions, not because the diagram looks better.

---

## Where agents fail

Per-step reliability compounds. With independent failure probability $p_i$ per
step:

$$
p(\varepsilon) = 1 - \prod_{i=1}^{n} (1 - p_i)
$$

At 95% per step, a 20-step task succeeds 36% of the time. Long autonomous
trajectories are not a prompt problem; they are an arithmetic problem.

- **No ground truth.** The agent cannot tell a correct answer from a plausible
  one, so it cannot know when to stop.
- **Cost.** Every step is a full-context call. A runaway loop is a bill.
- **Silent success.** The most dangerous outcome: the agent reports "done", the
  side effect never happened, and nothing raised.

---

## The engineering response

> Treat the agent as an untrusted process that occasionally produces good work.
> Everything it does is sandboxed, budgeted, and verified by code you wrote.

- **Sandbox** — a container, a scoped token, a read-only mount. Never a
  production credential, never `eval` on model output.
- **Budgets** — maximum steps, maximum tokens, wall-clock timeout, and a
  spending cap. All four, enforced in the loop, not in the prompt.
- **Verification** — a claim of success is not evidence. Assert the file exists,
  the test passes, the row was written.
- **Human approval** — required for irreversible actions. Draft, approve, then
  execute.

You already do this with your coding agent: it runs in a repository, its work
is a diff, and the tests decide. That is the pattern, and it generalises to
every agent you will build.
