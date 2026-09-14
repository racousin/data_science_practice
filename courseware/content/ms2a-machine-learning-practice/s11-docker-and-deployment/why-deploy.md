# From a Notebook to a Service

For ten sessions your models have lived in a notebook, and you have shipped
their predictions as a CSV file. Today the model itself leaves your laptop: it
becomes a small program that anyone can ask for a prediction, packaged so that
it runs the same on any machine.

<!-- notes: 12 minutes. The argument, not the tools. Most of the room has never
seen an API or Docker, so every word here should be one they already own:
notebook, file, function, "it works on my machine". Ask who has already had a
project fail on a teammate's laptop; there is always one. Check at the start
that Docker Desktop is installed and `docker run hello-world` works; anyone for
whom it does not starts the install now (next-but-one lesson), in the
background. -->

---

## A model nobody can use

You trained a classifier. It scores 0.98 on the test set. Now:

- a colleague wants a prediction for **one new flower**, from their app;
- the hospital's software wants a prediction **every time a form is saved**;
- your model has to keep answering **after you close your laptop**.

A notebook answers none of these. It runs on one machine, inside one Python
environment, when one person presses Shift+Enter.

---

## Batch or online

| | Batch | Online |
|---|---|---|
| What you send | a file of 10,000 rows | one row per request |
| When | once a day, once a week | the moment someone asks |
| Result | a predictions file | an answer in milliseconds |
| In this course | every ML-Arena CSV submission | today |

Both are "deployment". Online serving is the one that needs the two tools of
this session.

---

## Three things must leave the notebook

![From a notebook to a container any client can call](assets/deploy/deploy-path.png)

1. **The model**: the trained object, saved to a file (`model.joblib`).
2. **The API**: a small program that loads that file once and answers
   requests ("here are four measurements, which species?").
3. **The environment**: Python 3.13, scikit-learn 1.9.1, FastAPI, the
   operating system under them, all packed into a Docker **image**.

---

## Why an API

An API (Application Programming Interface) is **a function you call over the
network**.

- The caller does not need Python, scikit-learn, or your model file: it sends
  a few numbers and receives an answer.
- Anything can call it: a browser, a phone app, a spreadsheet macro, another
  model.
- You can retrain and replace the model behind it without the callers noticing,
  as long as the questions and the answers keep the same shape.

---

## Why Docker

Your API runs on your laptop. To run on a server, the server needs **exactly**
what your laptop has:

- the same Python version,
- the same 20 packages at the same versions (a model saved with one
  scikit-learn version and loaded with another prints a warning, or worse),
- the model file, and the command that starts the server.

Docker packs all of it into one **image**. Any machine with Docker runs that
image identically: no install instructions, no "it works on my machine".

---

## You have already used both

- Every agent you submitted to ML-Arena ran **inside a container**, on
  machines you never saw, with the packages the platform put in its image.
- Every call to `client.submit(...)` in the labs was **an API request** to
  ml-arena.com, and every leaderboard you looked at came back as the answer to
  one.

Today you build the other side of both.

---

## Where this session goes

1. Docker concepts: images, containers, registries.
2. Install Docker and run your first containers.
3. Workshop: an image-labelling platform, CVAT, started with one command.
4. APIs and HTTP, then a model served with FastAPI.
5. A Dockerfile that packs the model API into an image.
6. **Lab**: train a model, serve it, containerise it, use it from your browser,
   then ship the image.

---

## Check yourself

1. You submit a CSV of 5,000 predictions once. Batch or online?
   **Batch.** Online means one request, one answer, at the moment it is asked.
2. Name the three things that must leave the notebook for a model to be
   served. **The model file, the API that loads it, the environment it runs in.**
3. A teammate's laptop has scikit-learn 1.7, yours has 1.9.1. Which of the
   three is the problem, and which tool fixes it?
   **The environment; a Docker image that pins the version you trained with.**
