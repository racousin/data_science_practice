# The Coding-Assistant Landscape

There are a lot of these tools and they change every few months. What does not
change is the small number of shapes they come in — learn the shapes and any
individual tool takes an afternoon.

<!-- notes: 20 minutes. Resist doing a feature comparison; it is out of date
before the term ends. The taxonomy and the "what does it see" question are the
durable content. Ask the room what they already use — usually half of them. -->

---

## Three generations

![Three generations](/api/academic_courses/assets/lessons/178/generations.png)

The jump that matters is the third. An **agent** has a feedback loop: it runs
your tests, reads the failure, and tries again. Chat cannot — it never finds out
whether its answer worked.

Everything else about a tool is interface.

---

## Where the tools sit

![The map](/api/academic_courses/assets/lessons/178/assistant-map.png)


---

## Two axes that predict how a tool behaves


- **Terminal / CI, or editor UI.** A terminal tool can be scripted, run in a
  workflow, and used over SSH. An editor tool sees your cursor and your open
  files.
- **Proprietary product, or open source.** Determines whether you can run it
  against a model of your choosing, and whether it works with no network.

---

## The families

| Family | Examples | Shape |
|---|---|---|
| **Terminal agents** | Claude Code, Codex CLI, Gemini CLI | run in your repo, read/edit/execute, loop on tests |
| **Editor agents** | Cursor, Windsurf, Copilot agent mode | the same loop inside the IDE, with cursor context |
| **Inline completion** | Copilot, Codeium, Tabnine | next-lines suggestions as you type |
| **Open-source agents** | Aider, Continue.dev, OpenHands | same loop, you pick the model |
| **Local models** | Ollama + any of the above | no network, no account, weaker |
| **Review bots** | CodeRabbit, Copilot review, `gh` workflows | run on the pull request, comment on the diff |

<!-- notes: Say plainly that this table will be wrong within a year and the
column that will still be right is "Shape". -->

---

## What to actually compare

Not benchmark scores. Five questions that determine whether a tool fits your
work:

1. **What can it see?** One file, your open tabs, or the whole repository plus
   the output of commands it ran?
2. **Can it run things?** If not, it cannot verify, and you are back to chat.
3. **Where does your code go?** Some tools send the repository to a server.
   Read your employer's policy before you find out.
4. **What does it cost, and who pays?** Subscription, per-token API, or free.
5. **Can you leave?** A workflow built entirely on one product's interface does
   not transfer.

Question 3 is the one students underweight and employers do not.

---

## The economics, roughly

| Model | Typical shape |
|---|---|
| **Subscription** | a flat monthly fee, generous limits, easiest to reason about |
| **API / per token** | you pay for what you use; a long agent session is not free |
| **Free tiers** | real, useful, and rate-limited exactly when you are in a hurry |
| **Local** | free in money, expensive in time and quality |

Students: check [education.github.com](https://education.github.com) and the
student offers from the model vendors before paying for anything. Several are
free with a university email.

<!-- notes: Prices and tiers move constantly. Do not put numbers on a slide;
point at the pricing page. -->

---

## Model choice, in one paragraph

Every one of these tools is a harness around a model, and most let you switch.
The pattern that holds across vendors: a **large** model for design, unfamiliar
code and hard debugging; a **small, fast** one for mechanical edits, renames and
boilerplate. Claude Code's `/model` switches between Opus 5, Sonnet 5 and
Haiku 4.5 for exactly that reason.

Use the default until you have a measured reason to change it. "Bigger model"
is not a fix for an underspecified task.

---

## Open weights, running locally

Worth knowing this exists, and worth being honest about the gap.

```bash
# install ollama from https://ollama.com, then:
ollama pull qwen2.5-coder:7b
aider --model ollama/qwen2.5-coder:7b
```

- **For you:** no account, no network, no data leaving the machine. It runs on a
  laptop.
- **Against you:** materially weaker at multi-file work, and slow without a GPU.

The loop is identical; the quality of each step is not. If a local model is your
only option, keep the tasks small and read everything.

---

## What none of them do

- **Decide what to build.** The agent optimises the objective you name. Naming
  the wrong one is a Session 3 problem and no tool will save you from it.
- **Guarantee correctness.** *Plausible* and *correct* are different properties.
  These tools produce the first reliably and the second often.
- **Understand your problem better than you do.** They have your repository.
  They do not have the conversation you had with your supervisor.

---

## The rule for this course

> You are the author of every line you merge, whoever typed it.

Not a moral position — it is how the grading works. Your project is assessed on
repository quality and you will be asked to explain your code. "The agent wrote
it" is not an answer to "why is the learning rate scheduled here?".

Which is also why the previous lessons come first. A tool that edits your
repository is only safe on top of git, tests and review, and you now have all
three.

---

## What we do next

The rest of this session teaches **one** of these tools properly — Claude Code,
with Aider as the open alternative — because the loop is what transfers and you
learn a loop by running it, not by comparing it.

Four lessons: what changed, setup, the core loop, context, and guardrails. Then
a lab where an agent writes a feature and you have to defend the diff.
