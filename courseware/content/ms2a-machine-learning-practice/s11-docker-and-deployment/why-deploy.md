# Serving a Model: The Concepts

A trained model is a file. **Serving** a model means that a program loads this
file once and answers prediction requests from other programs, at any time.

<!-- notes: 20 minutes, no commands. One definition and one example per slide.
The six properties are the vocabulary of the whole workshop: each later step
points back to one of them (Step 1 pins the version = reproducibility, the
/health route = robustness, Render's cold start = latency, /docs = contract).
Concrete opening: a hospital form calls a model every time it is saved; a
notebook cell run by hand cannot do that. -->

---

## Reproducibility

The same input gives the same output on the laptop, the server and a
colleague's machine. This requires the same code, the same model file, the same
Python and package versions, and the same operating-system libraries.

Example: a `model.joblib` written by scikit-learn 1.6.1 in Colab, loaded with
scikit-learn 1.9.1 on a laptop:

```text
InconsistentVersionWarning: Trying to unpickle estimator
LogisticRegression from version 1.6.1 when using version 1.9.1.
This might lead to breaking code or invalid results.
Use at your own risk.
```

Steps 1 and 2 of this workshop pin `scikit-learn==1.6.1` on both sides.

---

## Robustness

A process crashes, a machine reboots. A **supervisor** restarts the process.

- [Docker](https://docs.docker.com/engine/containers/start-containers-automatically/):
  `--restart` takes `no` (default), `on-failure` (non-zero exit code),
  `always`, or `unless-stopped`.
- [Kubernetes](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#restart-policy):
  `restartPolicy` takes `Always`, `OnFailure` or `Never`.

A **health check** is an endpoint the supervisor calls at regular intervals,
here `GET /health`. On [Render](https://render.com/docs/health-checks), a check
succeeds on a 2xx or 3xx answer within 5 s; after 15 s of failed checks the
instance stops receiving traffic, after 60 s it is restarted.

---

## Scalability

![Clients reach several servers through one load balancer](assets/deploy/load-balancer.png)

*Figure: HFWMan, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Load_Balancing_Cluster_(NAT).svg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

- **Vertical** scaling: a bigger machine (more CPU, more memory).
- **Horizontal** scaling: more copies of the same service behind a **load
  balancer**, which spreads the requests across them
  ([Azure guide](https://learn.microsoft.com/en-us/azure/architecture/guide/design-principles/scale-out)).

Horizontal copies require a **stateless** API: no answer depends on a previous
request, and the model is loaded read-only. Order of magnitude: Cloudflare
handled 81 million HTTP requests per second on average in 2025
([Cloudflare Radar](https://blog.cloudflare.com/radar-2025-year-in-review/)).

---

## Batch or online

| | Batch | Online |
|---|---|---|
| Input | a file or table of many rows | one row per request |
| Trigger | a schedule or a job | a request, at any time |
| Output | a predictions file | one answer in the HTTP response |
| Latency | minutes to hours | milliseconds to seconds |
| Example | an ML-Arena CSV submission, a nightly scoring job | this workshop's `/predict` |

Azure ML names them [online and batch endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints);
SageMaker offers [real-time, serverless, asynchronous and batch](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
inference.

---

## Latency

**Latency** is the time between a request and its answer. It is a
distribution, described by **percentiles**: p50 (median), p95, p99, the value
below which 50 %, 95 %, 99 % of requests fall
([Google SRE book](https://sre.google/sre-book/service-level-objectives/)).

![Latency of the workshop API, local container and Render](assets/deploy/latency-local-vs-render.png)

A **cold start** is the first request after an idle period: the platform
starts the service before it answers. A Render free instance spins down after
15 minutes without traffic and takes about one minute to wake
([Render](https://render.com/docs/free)).

---

## Contract

The **contract** (interface) of an API is the set of field names, types,
units, answers and error codes it promises. The model behind it can be
retrained or replaced; the contract stays the same.

```text
POST /predict  {"bill_length_mm": 39.1, "bill_depth_mm": 18.7,
                "flipper_length_mm": 181, "body_mass_g": 3750}
200            {"species": "Adelie"}
422            when a field is missing or "bill_length_mm": "long"
```

FastAPI writes the contract as an [OpenAPI](https://spec.openapis.org/oas/latest.html)
document, served at `/openapi.json` and rendered at `/docs`.

---

## Check yourself

1. A model saved with scikit-learn 1.6.1 is loaded with 1.9.1. Which property
   is at stake? **Reproducibility: the package version is part of the
   environment the model depends on.**
2. Why must an API be stateless to scale horizontally? **The load balancer may
   send two requests of the same client to two different copies; each copy has
   to answer from the request alone.**
3. A service answers in 40 ms at p50 and in 60 s for the first request of the
   day. What explains the second figure? **A cold start: the instance was
   spun down and is started by that request.**
