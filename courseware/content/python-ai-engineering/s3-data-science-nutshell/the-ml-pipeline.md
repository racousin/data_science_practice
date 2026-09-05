# The ML Pipeline

A machine learning project is not "train a model". It is a pipeline where
training is one stage of seven, and rarely the one that decides the outcome.

<!-- notes: 30 minutes. The point to land: most of the failure modes are
upstream and downstream of the modelling, which is where students want to
spend all their time. -->

---

![The machine learning pipeline](assets/ds/pipeline.png)

---

## The stages

| # | Stage | Produces |
|---|---|---|
| 0 | Problem definition | a measurable objective and a success criterion |
| 1 | Data collection | a raw dataset, documented |
| 2 | Preprocessing & feature engineering | a modelling-ready dataset |
| 3 | Model selection & training | a trained model |
| 4 | Evaluation | an honest estimate of performance |
| 5 | Deployment | the model serving predictions |
| 6 | Monitoring | evidence it still works |

Steps 0 and 6 are where projects actually die. Steps 3 and 4 are where students
spend 90% of their time.

---

## Stage 0 — Problem definition

Before any code:

- What decision does this prediction inform?
- What does success look like, **as a number**?
- Is machine learning the right tool, or would a rule do?
- What is the baseline you must beat?

If you cannot answer the last one, you cannot tell whether your model is good.

---

## The three questions that frame a problem

1. **What is `X`?** What is available at prediction time — genuinely available,
   not available in the historical table.
2. **What is `y`?** What exactly are you predicting, and is it observable?
3. **What is the metric?** The single number that decides between two models.

Getting question 1 wrong is *leakage*, and it is the most common way to produce
a model that scores 0.99 and is worthless.

---

## Stage 1 — Data collection

Where the data comes from constrains everything downstream. Record:

- **Source and date.** A dataset with no provenance cannot be reproduced.
- **Sampling.** Who or what is *not* in this data?
- **Licence.** Especially for anything you deploy.

Version your data the way you version code. A result you cannot reproduce
because the data moved is not a result.

---

## Stage 2 — Preprocessing & feature engineering

Cleaning, encoding, scaling, deriving. This is usually the largest share of the
work.

The rule that matters more than any technique:

> Every transformation must be **fit on the training set only** and then applied
> to validation and test.

A scaler fitted on the full dataset has already shown your model the test set's
mean. The score you report is then optimistic, and you find out in production.

---

## Stage 3 — Model selection and training

Start with the simplest thing that could work:

| Data | First model |
|---|---|
| Tabular | linear / logistic regression, then gradient boosting |
| Images | a pretrained CNN, fine-tuned |
| Text | a pretrained transformer, fine-tuned |
| Sequential decisions | a scripted policy, then RL |

The simple model is not a formality. It is the number the complicated model has
to beat to justify itself.

---

## Stage 4 — Evaluation

Covered in depth in the next two lessons. The one-line version:

> The score you optimise against must be computed on data the model has never
> influenced.

---

## Stage 5 — Deployment

A model that lives in a notebook has no users. Deployment forces questions the
notebook let you avoid: latency budget, input validation, what happens when a
feature is missing.

For this course, "deployment" is a submission to ML-Arena: your code runs on
inputs you have never seen, in an environment you did not configure. That is a
genuine deployment discipline, at student scale.

---

## Stage 6 — Monitoring

Models decay. The world moves, the input distribution shifts, and yesterday's
accuracy stops holding.

- **Data drift** — the inputs change distribution
- **Concept drift** — the relationship between `X` and `y` changes
- **Pipeline breakage** — an upstream schema changes and nobody tells you

---

## Baseline and iterate

The single most useful habit in this lesson:

1. Build the **simplest end-to-end pipeline** that produces a submission —
   even if the model predicts the mean.
2. Measure it.
3. Improve one thing. Measure again.

A complete bad pipeline is worth more than an excellent model with no pipeline
around it. It tells you where the effort should go, and it means you always have
something to submit.

<!-- notes: Tie this to the project: teams that build the pipeline in week one
outperform teams that spend three weeks on the model. Every year. -->

---

## Where the time actually goes

| Stage | Typical share |
|---|---|
| Problem definition | 5% |
| Data collection | 15% |
| Preprocessing & features | 40% |
| Modelling | 15% |
| Evaluation | 10% |
| Deployment & monitoring | 15% |

Budget accordingly — including in Lab 3, where Parts A and B cost you ten of
your forty-five minutes before you have trained anything.

---

## Check yourself

1. Which two stages does this lesson say projects actually die at, and which two
   swallow 90% of a student's time?

   **Answer.** Stage 0 (problem definition) and stage 6 (monitoring) are where
   projects die; stages 3 (training) and 4 (evaluation) are where the time goes.

2. Of the three framing questions, getting exactly one of them wrong is called
   *leakage*. Which one, and what does the resulting model look like?

   **Answer.** Question 1, "what is `X`?" — using something that is in the
   historical table but not genuinely available at prediction time. It produces a
   model that scores 0.99 and is worthless.

3. Run this. It is Stage 2's rule — *fit on the training set only* — made
   visible. You should get exactly the output shown.

   ```python
   import numpy as np
   from sklearn.preprocessing import StandardScaler
   train, test = np.array([[1.0], [2.0], [3.0]]), np.array([[100.0]])
   print(StandardScaler().fit(train).transform(test).round(2))                    # -> [[120.02]]
   print(StandardScaler().fit(np.vstack([train, test])).transform(test).round(2))  # -> [[1.73]]
   ```

   **Answer.** The second scaler was fitted on the full dataset, so it has
   already seen the test point: an extreme value is rescaled to an ordinary
   1.73 and stops looking extreme. That is the optimistic score you only find
   out about in production.

4. You have four hours and a dataset you have never seen. What does "baseline
   and iterate" tell you to build first, and why that rather than the model?

   **Answer.** The simplest end-to-end pipeline that produces a submission, even
   if the model predicts the mean. A complete bad pipeline tells you where the
   effort should go, and it means you always have something to submit.
