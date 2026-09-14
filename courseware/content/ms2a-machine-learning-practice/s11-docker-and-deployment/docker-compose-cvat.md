# Workshop — Label Images with CVAT and Docker Compose

Supervised learning needs labels, and someone has to draw them. CVAT is an
open-source annotation platform used to build detection and segmentation
datasets. Installed by hand it means setting up ten separate services that
must find each other; with Docker Compose it is three commands. You install
it, label a few images, and export a dataset.

<!-- notes: 30 minutes: 5 on Compose, 25 hands-on. Everything here was run on
2026-09-14 with CVAT v2.75.0 on an Apple M4 (OrbStack). The download is
1.44 GB and the images take 5.74 GB on disk: students must run the
"Before the session" slide at home, or the room's network decides the
workshop. CVAT runs about 4 GB of RAM when idle; an 8 GB laptop copes if
little else is open. First start: about 40 s after `up -d` returns before the
login works; students who see "404 page not found" have simply been too quick.
CVAT releases every two weeks: keep the pinned tag, whatever is newer. -->

---

## Before the session (at home, 10 minutes of download)

```bash
git clone --depth 1 --branch v2.75.0 https://github.com/cvat-ai/cvat
cd cvat
docker compose pull
```

- **1.44 GB** to download, **5.74 GB** on disk once unpacked.
- `--branch v2.75.0` pins a release. A plain `git clone` gets the development
  branch and its development images.
- Use **Google Chrome**: it is the only browser CVAT supports.
- Port **8080** must be free: `docker rm -f web` if the nginx of the previous
  lesson still runs.

---

## One application, many containers

![The 18 containers CVAT starts, grouped by role](assets/deploy/cvat-services.png)

A real application is rarely one program. CVAT is a web page, an API server,
8 background workers, a database, two caches, a permission service, a
logging pipeline and a proxy in front of all of them: **18 containers**, from
10 different images.

Starting them one `docker run` at a time, with the right ports, volumes,
networks, environment variables and start order, is where installs go wrong.

---

## Docker Compose

**Docker Compose** describes the whole application in one file,
`docker-compose.yml`, and starts or stops all of it with one command. An
excerpt of CVAT's:

```yaml
services:
  cvat_db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: cvat
    volumes:
      - cvat_db:/var/lib/postgresql/data
  traefik:
    image: traefik:v3.6
    ports:
      - 8080:8080
volumes:
  cvat_db:
```

Every key is a `docker run` flag you already know: `image`, `ports` (`-p`),
`volumes` (`-v`), `environment` (`-e`). Containers of one Compose file share a
network and reach each other by service name: the server connects to the
host `cvat_db`.

---

## Compose commands

Run them in the folder that holds `docker-compose.yml`:

| Command | What it does |
|---|---|
| `docker compose up -d` | create and start every service, in the background |
| `docker compose ps` | the application's containers and their state |
| `docker compose logs -f cvat_server` | follow one service's output |
| `docker compose stop` | stop everything, keep the containers |
| `docker compose down` | stop and remove the containers; **volumes are kept** |
| `docker compose down -v` | the same, **and delete the volumes: all data** |

---

## Step 1: start it (about 3 minutes)

```bash
cd cvat
docker compose up -d
docker compose ps
```

- Measured on a first run: **2 min 03 s** for `up -d` with nothing cached,
  then **39 s** more before the login page works.
- Until then the browser shows `404 page not found`, then a grey
  *Connecting...* spinner. Wait, then reload.
- On an Apple silicon Mac every `up` prints `The requested image's platform
  (linux/amd64) does not match the detected host platform` for the CVAT
  images: they exist only for amd64 and run under emulation. It works.

---

## Step 2: create your account

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Username, email, password, twice. `docker exec` runs Django's own admin
command **inside** the running server container: nothing Python-related is
installed on your laptop.

Then open **http://localhost:8080** in Chrome and sign in: type the username,
press **Next**, then the password. You land on the task list.

Use `localhost`: CVAT answers `404 page not found` on `127.0.0.1:8080`.

---

## Step 3: create a task

![Create a task: a name, three labels, eight images](assets/deploy/cvat-create-task.png)

**+** → **Create a new task**:

1. **Name**: anything.
2. **Labels**, *Constructor* tab: **Add label**, type a name, **Continue**.
   Three labels are plenty (`cup`, `phone`, `pen`, or whatever is in your
   images).
3. **Select files** → *My computer*: about ten images of your own, phone
   photos of your desk will do.
4. **Submit & Open**. Eight images took 16 s.

---

## The task and its job

![A task holds the labels and one job over all the images](assets/deploy/cvat-task-jobs.png)

A **task** holds the images and the label set; a **job** is a slice of it
assigned to one annotator. With a team, you split a task into jobs; alone, one
job covers all the frames. Click **Job #1**.

---

## Step 4: draw boxes

![Choose the rectangle tool, a label, and 2 Points](assets/deploy/cvat-draw-rectangle.png)

1. Hover the **rectangle** icon in the left toolbar.
2. Choose the **Label**, the method **2 Points**, then **Shape**.
3. **Click** one corner of the object, then **click** the opposite corner.
   Click, move, click: dragging places only the first corner.

---

## Label, save, move on

![Three boxes on one frame, listed on the right](assets/deploy/cvat-annotate.png)

- Every box appears in the **Objects** list on the right, where you can change
  its label.
- **Save** (or `Ctrl+S` / `Cmd+S`). Nothing is stored until you do.
- The arrows at the top move between frames. Label at least three images.

---

## Step 5: export the dataset

**Back to tasks** → your task → **Actions** → **Export task dataset**:

![Export format COCO 1.0, images not included](assets/deploy/cvat-export.png)

- **Export format**: `COCO 1.0`, then again with `YOLO 1.1`.
- *Save images* is **off** by default: the export holds the labels only.
- **OK** starts the export in the background. Nothing downloads yet.

---

## Step 6: download it

Top menu **Requests** → the finished export → **⋮** → **Download**:

![The Requests page lists finished exports](assets/deploy/cvat-requests.png)

The file name contains a space (`task_1_annotations_..._coco 1.0.zip`):
quote it in the terminal, `unzip "task_1_annotations_...coco 1.0.zip"`.

---

## What you exported

**COCO 1.0**: one JSON file, `annotations/instances_default.json`:

```json
"categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "person"}, ...],
"images": [{"id": 1, "width": 512, "height": 512,
            "file_name": "astronaut.jpg"}, ...],
"annotations": [{"image_id": 1, "category_id": 2,
                 "bbox": [91.52, 16.28, 306.16, 493.76]}, ...]
```

`bbox` is `[x, y, width, height]` in pixels, from the top-left corner.

**YOLO 1.1**: one text file per image, one line per box:

```text
1 0.477734 0.513984 0.597969 0.964375
```

`class centre_x centre_y width height`, divided by the image size:
(91.52 + 306.16 / 2) / 512 = 0.4777. The format Session 6's detectors train on.

---

## Stop, restart, clean up

| You run | Containers | Your tasks and labels |
|---|---|---|
| `docker compose stop` | stopped | kept |
| `docker compose up -d` again | running in about 30 s | kept |
| `docker compose down` | removed | **kept**, in 7 named volumes |
| `docker compose down -v` | removed | **deleted**, account included |

`docker volume ls` lists them: `cvat_cvat_db`, `cvat_cvat_data`, ... The data
lives in volumes, not in containers, which is why deleting every container
loses nothing.

---

## When it does not work

| Symptom | Fix |
|---|---|
| `404 page not found` right after `up -d` | wait 40 s and reload |
| `404 page not found` on `127.0.0.1:8080` | use `http://localhost:8080` |
| `Bind for 0.0.0.0:8080 failed: port is already allocated` | free 8080 (`docker ps` shows who holds it), then `docker compose down` **and** `up -d`: a plain `up -d` leaves the proxy without its port |
| `cannot attach stdin to a TTY-enabled container` | run `createsuperuser` in a real terminal, not from a script or notebook |
| `Invalid username/password` after `down -v` | the volumes, and your account, are gone: create it again |
| the laptop slows to a crawl | CVAT uses about 4 GB of RAM: `docker compose stop` when done |

---

## What Compose bought you

Installed by hand, CVAT needs PostgreSQL, Redis, Kvrocks, ClickHouse, Grafana,
Open Policy Agent, Vector, Traefik, a Python backend with its system libraries, and a
built JavaScript front end, each configured to find the others.

With Compose:

- **install**: `docker compose up -d`
- **same result on every laptop** in the room, and on a server
- **uninstall without traces**: `docker compose down -v`, then `docker rmi`
  the images

That is the argument for Docker in one workshop. The next lessons turn it
around: you build the image yourself.

---

## Check yourself

1. You ran `docker compose down` and then `up -d`. Are your labels still
   there? **Yes: `down` removes containers, and the labels live in named
   volumes.**
2. In CVAT's compose file, how does the server find the database?
   **By service name: the host `cvat_db`, on the network Compose creates.**
3. A COCO box is `[40, 60, 100, 50]` on a 200 × 200 image. What is its YOLO
   line for class 0? **`0 0.45 0.425 0.5 0.25`**: centre (90, 85) and size
   (100, 50), each divided by 200.
4. Which single command deletes everything you labelled?
   **`docker compose down -v`.**
