# Step 5 — An HTML Page for the Model

A browser page is one more client of `POST /predict`. The same FastAPI app
serves it at `/`, so the image and the Render service of Step 4 carry both.

<!-- notes: 20 minutes. Tested on 2026-10-19 with the reference penguin-api:
the form returns Adelie, Chinstrap and Gentoo for the three birds. Where
students get stuck: (1) the page opened as a file (file:///.../index.html)
instead of through the server: fetch("/predict") then has no server to reach;
(2) index.html added to main.py but not to the Dockerfile's COPY line: the
container answers 500 on `/` with a missing-file error in its logs; (3) a
missing `name=` attribute on an input: that field is absent from the body and
the API answers 422, and the page shows `undefined`; (4) Render still serving
v1 because the Image URL was not changed to :v2. The browser's developer tools
(Network tab) show the request and the 422 body. -->

---

## A page is an HTTP client

| Client | Request it sends |
|---|---|
| `curl` | `POST /predict`, JSON typed by hand |
| `/docs` | `POST /predict`, JSON from the form of the docs page |
| `index.html` | `POST /predict`, JSON built from 4 input fields |

Same endpoint, same `Penguin` contract, same answer. Only the caller changes.

---

## `index.html`: the form

Four inputs, one button, one paragraph for the answer. One input:

```html
<label>Bill length (mm)
  <input name="bill_length_mm" type="number" step="any" value="39.1">
</label>
```

The complete page: [`penguin-api/index.html`](https://github.com/racousin/data_science_practice/tree/main/website/public/modules/ms2a-machine-learning-practice/deploy/penguin-api/index.html). The other three follow the same pattern, with `name` equal to the API field
(`bill_depth_mm`, `flipper_length_mm`, `body_mass_g`). The form is
`<form id="form">` and ends with `<button>Predict</button>`; the answer goes
in `<p id="out"></p>`.

---

## `index.html`: the script

In a `<script>` at the end of the page, inside
`document.getElementById("form").onsubmit = async (event) => { ... }`, after
`event.preventDefault();` (which keeps the browser from reloading the page):

```js
const data = Object.fromEntries(new FormData(event.target));
const r = await fetch("/predict", {
  method: "POST", body: JSON.stringify(data),
  headers: {"Content-Type": "application/json"}});
document.getElementById("out").textContent = (await r.json()).species;
```

- `FormData` reads every input with a `name` into a dict-like object.
- Form values are strings (`"46.1"`). Pydantic converts a numeric string to
  `float`, so the `Penguin` contract accepts them.

---

## Serving the page from FastAPI

Two additions to `main.py`:

```python
from fastapi.responses import FileResponse

@app.get("/")
def home(): return FileResponse("index.html")
```

- `GET /` now returns the page; `fetch("/predict")` then calls the same host
  and port that served it.
- Page and API share one **origin** (scheme + host + port), so CORS does not
  apply. **CORS** is the rule by which a browser blocks a page's requests to
  another origin unless that server allows them
  ([MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS)).

**Checkpoint:** with `uv run fastapi dev main.py`, `http://localhost:8000/`
shows the form, and **Predict** prints `Adelie` for the default values.

---

## The page in the image

The Dockerfile's code line gains the page:

```dockerfile
COPY main.py model.joblib index.html ./
```

```bash
docker build -t penguin-api . && docker run --rm -p 8000:8000 penguin-api
```

Only the last two layers are rebuilt; `uv sync` comes from the cache.

![The form at http://localhost:8000/ after Predict, for the Gentoo bird](assets/deploy/html-form.png)

---

## Version 2 on Render

```bash
docker build --platform linux/amd64 -t USER/penguin-api:v2 .
docker push USER/penguin-api:v2
```

Then, on Render, the steps of Step 4's "New version" section: **Settings →
Image → Source → Edit** → `docker.io/USER/penguin-api:v2` → **Connect**. A new tag
guarantees that Render pulls the new image.

**Checkpoint:** on `https://<name>.onrender.com/`, the three birds typed in the
form give `Adelie`, `Chinstrap`, `Gentoo`.

---

## Check yourself

1. `index.html` opened by double-click (`file:///…`) shows the form, but
   **Predict** does nothing. Why? **`fetch("/predict")` is relative to the
   page's origin; a file has no server behind it. The page must be loaded from
   the API's URL.**
2. Why does this page need no CORS configuration? **The page and `/predict`
   come from the same origin: same scheme, host and port.**
3. The form sends `"46.1"`, a string. Why no `422`? **Pydantic converts a
   numeric string to the `float` the `Penguin` contract declares.**
