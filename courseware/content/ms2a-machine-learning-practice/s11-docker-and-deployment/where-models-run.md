# Where Models Run: On-Premise, Cloud, Managed

A served model runs on a machine that someone owns, operates and pays for. The
options differ in one respect: how much of the stack the organisation manages
itself, and how much a provider manages.

<!-- notes: 10 minutes. The responsibility diagram is the slide to spend time
on: read one column top to bottom, then compare IaaS and PaaS. Render, used in
Step 4, is a PaaS. The Linux slide justifies `--platform linux/amd64` in
Step 3; the roadmap slide is the plan for the next two and a half hours. -->

---

## On-premise

The organisation owns the machines, in its own building or data centre: a
hospital's servers, a bank's cluster, a university's GPU cluster.

- It controls where the data is stored and who can reach it.
- It buys the hardware, installs the operating system, applies updates and
  replaces failed disks.

---

## Cloud virtual machines (IaaS)

**Infrastructure as a Service**: a virtual machine rented from a provider,
started in minutes and billed for the time it runs.

- [AWS EC2](https://aws.amazon.com/ec2/)
- [Google Compute Engine](https://cloud.google.com/products/compute)
- [Azure Virtual Machines](https://azure.microsoft.com/en-us/products/virtual-machines)

The provider manages the hardware and the network; the customer manages the
operating system, Python, the packages and the API.

---

## Platforms (PaaS) and managed ML endpoints

**Platform as a Service**: the customer gives a container image or a model,
and the platform runs it, restarts it and gives it a public URL.

- [Render](https://render.com/), used in Step 4: runs a Docker image as a web
  service.
- [Amazon SageMaker AI](https://aws.amazon.com/sagemaker/ai/deploy/),
  [Azure Machine Learning endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints),
  Google [Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions)
  (formerly Vertex AI): managed endpoints for online and batch inference.

---

## Who manages what

![Customer-managed and provider-managed layers for on-premise, IaaS, PaaS and SaaS](assets/deploy/onprem-iaas-paas-saas.png)

*Figure: P. Rosati and T. Lynn (2020), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Comparison_of_on-premise,_IaaS,_PaaS,_and_SaaS.png), public domain.*

From left to right, the provider manages more layers and the customer fewer.
In this workshop the application is the model API; Render (PaaS) manages
everything below it.

---

## The operating system: Linux

Unix-like systems run **92.2 %** of the websites whose operating system is
known, Windows 8.0 % ([W3Techs](https://w3techs.com/technologies/overview/operating_system),
19 October 2026).

A container image holds a Linux userland, built for one processor
architecture. A laptop with an ARM processor (Apple M-series) builds ARM
images by default; Render requires `linux/amd64`
([Render docs](https://render.com/docs/deploy-an-image)). Step 3 therefore
builds with:

```bash
docker build --platform linux/amd64 -t USER/penguin-api:v1 .
```

---

## The workshop path

![Colab, laptop API, Docker image, Docker Hub, Render, HTML page](assets/deploy/workshop-path.png)

| Step | Where | Result |
|---|---|---|
| 1 | Colab | `model.joblib` |
| 2 | laptop | a FastAPI app answering `/predict` |
| 3 | laptop | a Docker image of the app |
| 3 | Docker Hub (registry) | the image, public |
| 4 | Render (cloud) | the image served at `https://<name>.onrender.com` |
| 5 | browser | an HTML page calling `/predict` |

---

## Check yourself

1. A hospital requires patient data never to leave its building. Which option
   meets this requirement? **On-premise.**
2. On a cloud VM (IaaS), who installs Python and the packages? **The customer.
   The provider manages the hardware, the network and the virtualisation.**
3. Why does Step 3 pass `--platform linux/amd64`? **The image must match the
   server's operating system and processor: Linux on `amd64`, whatever the
   laptop runs.**
