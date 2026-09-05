# Setup

Two tracks. Everyone does track A if they can; track B exists so that nobody is
blocked by an account or a budget.

<!-- notes: 20 minutes, hands on keyboards. Walk the room. The students who get
stuck get stuck on Node or on auth, not on the concepts. -->

---

## Track A — Claude Code

A terminal agent from Anthropic. Runs where your code is, uses your shell, edits
your files.

`npm` ships with Node.js. If `node --version` prints nothing you have two
routes, and the second needs no Node at all:

```bash
npm install -g @anthropic-ai/claude-code          # needs Node.js 18+
curl -fsSL https://claude.ai/install.sh | bash    # macOS / Linux, native
```

Then, from inside a project directory:

```bash
cd my_project
claude
```

The first run walks you through authentication. It also matters that you start
it **inside** the repository — the working directory defines what it can see.

---

## Also available

The same tool is reachable from more than a terminal:

- **Terminal** (`claude`) — what we use in class
- **IDE extensions** — VS Code and JetBrains
- **Desktop app** — macOS and Windows
- **Web** — [claude.ai/code](https://claude.ai/code)

Pick one. The mental model in this session is identical across all of them.

---

## First contact

In your Session 1 repository, `git status` must be clean first, and:

```bash
git switch -c agent-sandbox
```

so that nothing here lands on `main`. *Guardrails & Review* explains why; do it
now anyway — the second prompt below edits your files.

Then try these, in order, on that branch:

```text
> what does this project do?

> add a docstring to every public function in src/

> run the tests
```

Notice what happens between your message and the answer: it reads files, greps,
runs commands. You are watching the loop.

---

## Useful from minute one

| Command | Effect |
|---|---|
| `/init` | Generate a `CLAUDE.md` describing the project |
| `/clear` | Wipe the conversation, keep the session |
| `/model` | Switch model (Opus 5 / Sonnet 5 / Haiku 4.5) |
| `/config` | Theme and behaviour settings |
| `#` prefix | Save the line to memory |
| `!` prefix | Run a shell command directly in the session |
| `Esc` | Interrupt — use it early and often |
| `/help` | Everything else |

`Esc` is the one to internalise today. Stopping a wrong direction after ten
seconds costs nothing; letting it finish costs a review.

---

## Model choice

Claude Code runs on the Claude model family — currently **Opus 5** (the default
for hard work), **Sonnet 5**, and **Haiku 4.5** for fast, cheap operations.
`/model` switches between them; `/config` holds everything else.

For this course the default is correct. Change it when you have a measured
reason, not a hunch.

---

## Track B — an open alternative

If you have no Claude access, use **Aider**. It is open source, model-agnostic,
and implements the same loop: read repository, propose edit, apply as a commit.

```bash
uv tool install aider-chat
cd my_project
aider
```

Aider commits every change it makes, which makes `git diff` and `git revert`
your review interface.

---

## Fully local, no API

Aider can drive a local open-weights model through Ollama. Slower and less
capable, but it runs on your laptop with no account and no network:

```bash
# install ollama from https://ollama.com, then:
ollama pull qwen2.5-coder:7b
aider --model ollama/qwen2.5-coder:7b
```

Expect to supervise this much more closely. The loop is the same; the quality of
each step is not.

<!-- notes: Be straight about the gap. Overselling the local option produces
students who conclude the whole technique is useless. -->

---

## Comparison

| | Claude Code | Aider | Aider + Ollama |
|---|---|---|---|
| Cost | subscription / API | API for the model | free |
| Network | required | required | not required |
| Repo-wide navigation | strong | good | weak |
| Runs your tests | yes | yes | yes |
| Auto-commits | no (you commit) | yes, per change | yes, per change |

---

## Before you continue

Confirm all four:

- [ ] The tool starts inside your Session 1 repository.
- [ ] It can answer "what does this project do?" using your actual files.
- [ ] It can run `pytest` and report the result.
- [ ] You have interrupted it at least once with `Esc` / `Ctrl-C`.

The fourth one is not a joke. Students who have never interrupted an agent let
bad runs finish.

---

## Check yourself

1. You start `claude` from your home directory instead of from inside the
   repository. What does that cost you?

   **Answer.** The working directory defines what the agent can see, so it
   cannot read your project's files. `cd` into the repository first.

2. Run this in the repository you just used for *First contact*. You should get
   exactly the output shown.

   ```bash
   git branch --show-current   # -> agent-sandbox
   ```

   **Answer.** If it prints `main`, the second prompt above edited your graded
   Session 1 work in place, with nothing to `git switch` back to. Branch first,
   every time.

3. You have no Claude access and no network. Which column of the comparison
   table still works, and what do you give up?

   **Answer.** Aider + Ollama: free and network-not-required. You give up
   repo-wide navigation (weak), and each step is lower quality, so you
   supervise much more closely.

4. Which command switches model, and which key stops a run that has gone the
   wrong way?

   **Answer.** `/model` switches between Opus 5, Sonnet 5 and Haiku 4.5.
   `Esc` interrupts — the lesson's "one to internalise today".
