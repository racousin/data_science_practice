# Step 4 — Serving the Image on Render

The image `USER/penguin-api:v1` is on Docker Hub. Render pulls it, runs it on
one of its servers, and gives it a public HTTPS address.

<!-- notes: 20 minutes. Dry run on 2026-10-19: racousin90/penguin-api:v1 on a
Free instance in Frankfurt, live at https://penguin-api-y9gh.onrender.com
2 min 50 s after "Deploy web service" (image pull ~25 s, Python imports ~31 s
on 0.1 CPU, then one restart for the port, see the logs slide). No card was
asked for on that account; Render has been reported to ask some new accounts
for a card check, so students without a card pair with one who has one. Where
students get stuck: (1) the $7 plan is selected by default; (2) an arm64 image
(no --platform linux/amd64): the deploy fails with `exec format error`;
(3) a private Docker Hub repository: Render needs a credential, the public one
does not; (4) the URL is not https://<name>.onrender.com when the name is
taken: Render appends a suffix, the service page shows the real one. -->

---

## What Render is

A **platform as a service** (PaaS): it runs a container image and handles
the machine, the operating system, the HTTPS certificate and the address.

| Free instance | |
|---|---|
| Compute | 0.1 CPU, 512 MB RAM, $0 |
| Idle | spins down after 15 min without a request |
| Cold start | the next request waits for the container to start (~1 min) |
| Quota | 750 instance-hours per month per workspace |

Source: [render.com/docs/free](https://render.com/docs/free). The idle and
cold-start rows are the latency concepts of the first lesson, seen live.

---

## A new web service

A Render account (render.com, sign-up with GitHub or e-mail), then
**+ New → Web Service**, or **Create new service** in a project:

![An empty Render project and its Create new service button](assets/deploy/render-project.png)

---

## A web service

![The service types; the API is a Web Service](assets/deploy/render-service-types.png)

A **Web Service** receives HTTP requests from the internet. A *Static Site*
serves files only; a *Background Worker* receives no requests.

---

## Source: an existing image

![Source Code: Existing Image, with an Image URL field](assets/deploy/render-existing-image.png)

- **Existing Image** → Image URL: `docker.io/USER/penguin-api:v1` → **Connect**.
- The repository is public: no credential.
- Render accepts only `linux/amd64` images
  ([deploying an image](https://render.com/docs/deploying-an-image)).

---

## Name, region, instance type

![The form: name penguin-api, region Frankfurt, the $0 Free instance selected](assets/deploy/render-config.png)

- **Name**: `penguin-api`. The address becomes
  `https://penguin-api-<suffix>.onrender.com` when the name is already taken.
- **Region**: Frankfurt (EU Central), the closest to France.
- **Compute**: the form opens with **$7 / month** selected. **$0 / month,
  Free** is the first row.

Then **Deploy web service**.

---

## The logs

![Render's logs: FastAPI starts, the port 8000 is detected, the deploy restarts](assets/deploy/render-logs-port-detect.png)

- The same Uvicorn lines as on the laptop: `Uvicorn running on
  http://0.0.0.0:8000`.
- Render expects port 10000 by default. It detected 8000 itself, then
  restarted once (`New primary port detected: 8000`). An environment variable
  `PORT` = `8000` in the form avoids that restart
  ([port binding](https://render.com/docs/web-services#port-binding)).
- Python imports took 31 s on 0.1 CPU, against under 2 s on the laptop.

---

## Testing it

![The service page: Free, Live, image racousin90/penguin-api:v1 and its URL](assets/deploy/render-service.png)

```bash
curl https://penguin-api-<suffix>.onrender.com/health
curl -X POST https://penguin-api-<suffix>.onrender.com/predict \
  -H "Content-Type: application/json" -d '{"bill_length_mm": 39.1, ...}'
```

`/docs` works in the browser too. Measured: **~100 ms** per warm request
(laptop container: 1.5 ms) and **COLD s** for the first request after 17
idle minutes.

**Checkpoint:** `/health` answers `{"status":"ok"}` on the public URL, from a
phone.

---

## A new version

```bash
docker build --platform linux/amd64 -t USER/penguin-api:v2 .
docker push USER/penguin-api:v2
```

On Render: **Settings → Image URL** → `docker.io/USER/penguin-api:v2` →
**Save**, then **Manual Deploy → Deploy latest reference**. Render does not
watch Docker Hub; a [deploy hook](https://render.com/docs/deploy-hooks) is
the scripted equivalent.

---

## Check yourself

1. The first request of the morning takes about a minute, the next ones
   ~100 ms. Why? **The Free instance spun down after 15 idle minutes; the
   first request waits for the container to start (cold start).**
2. The deploy fails with `exec format error`. What happened? **The image was
   built for arm64 (a Mac with an M chip, no `--platform linux/amd64`); Render
   runs amd64 only.**
3. Why a new tag `v2` instead of re-pushing `v1`? **Render deploys the
   reference it is given and caches public images; a new tag names exactly the
   new image.**
