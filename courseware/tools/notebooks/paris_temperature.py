"""Paris Temperature Forecast — the minimal Colab notebook.

Load the archive, write `agent.py` (its `features()` is imported back, so the
model trains on exactly what the agent computes live), fit one linear model
per lead time, turn its residuals into an ensemble, score it over 2025-2026
the way the challenge does, and submit. The challenge's env is
`tests-dummy/catalog/paris_temperature/env.py` in the platform repo: the
`predict(rows)` contract and the CRPS here mirror it.

Writes one file; safe to run:

    uv run python courseware/tools/notebooks/paris_temperature.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "website" / "public" / "modules" / "ms2a-machine-learning-practice" / "challenges"
NAME = "paris-temperature-forecast.ipynb"
GITHUB = "racousin/data_science_practice"
COLAB = (f"https://colab.research.google.com/github/{GITHUB}/blob/main/"
         f"{OUT.relative_to(REPO)}/{NAME}")
DATA_BASE = f"https://raw.githubusercontent.com/{GITHUB}/main/{OUT.relative_to(REPO)}/"
ARCHIVE = "weather_europe_2020_2026.csv.gz"
# The live feed since the archive ends (built by build_live_snapshot.py in the
# platform repo, tests-dummy/catalog/paris_temperature/).
LIVE = "weather_europe_live_2026-03-04_2026-11-04.csv.gz"
CHALLENGE_ID = 203

md, code = nb.md, nb.code


def build() -> dict:
    cells = [
        md(
            "# Paris Temperature Forecast — a minimal agent",
            "",
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})",
            "",
            f"Challenge: https://ml-arena.com/viewchallenge/{CHALLENGE_ID}",
            "",
            "Every 6 hours your agent receives the last 72 hours of weather in 45",
            "European cities, as rows of the archive CSV, and returns plausible Paris",
            "temperatures 6 h and 48 h later. This notebook trains a linear model,",
            "scores it like the challenge, and submits it.",
            "",
            "Submitting needs one Colab secret, `MLARENA_API_KEY` (ML-Arena → Profile →",
            "API Keys): the key icon on the left, then allow this notebook to read it.",
        ),
        code(
            "import os, sys, subprocess",
            'if "google.colab" in sys.modules:',
            '    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlarena-sdk"], check=True)',
            "",
            "import json",
            "import numpy as np",
            "import pandas as pd",
            "",
            f'BASE = "{DATA_BASE}"',
            "# Open-Meteo archive 2020-01 → 2026-03, then the live feed (OpenWeatherMap) since:",
            "# same columns, contiguous hours, so the two simply stack.",
            f'df = pd.concat([pd.read_csv(BASE + "{ARCHIVE}"), pd.read_csv(BASE + "{LIVE}")],',
            "               ignore_index=True)",
            "df.tail()",
        ),
        md(
            "## 1. What `predict` receives",
            "",
            "The rows are sorted by `timestamp` then `city_name`, 45 per hour, so the",
            "72 hours ending at hour `k` are rows `45*(k-71)` to `45*(k+1)`. The",
            "challenge sends exactly this, as a list of dicts:",
        ),
        code(
            "N_CITIES, HOURS = 45, 72",
            "",
            "def rows_at(k):",
            '    """The rows predict() gets when hour k is the last observed hour."""',
            '    return df.iloc[N_CITIES * (k - HOURS + 1): N_CITIES * (k + 1)].to_dict("records")',
            "",
            "rows = rows_at(30_000)",
            "pd.DataFrame(rows).tail(3)",
        ),
        md(
            "## 2. The agent",
            "",
            "`features()` turns the frame into one row per hour: every city's",
            "temperature now and its change over 24 h, Paris's recent past and",
            "weather, the hour and the season. The model is a linear regression per",
            "lead time; its ensemble is the prediction plus the quantiles of its",
            "errors on a held-out year. The coefficients go to `model.json`.",
        ),
        code(
            "%%writefile agent.py",
            "import json",
            "import os",
            "",
            "import numpy as np",
            "import pandas as pd",
            "",
            "",
            "def features(df):",
            '    """One row per timestamp (needs 49 hours of history)."""',
            '    df = df.assign(timestamp=pd.to_datetime(df["timestamp"]))',
            '    temp = df.pivot(index="timestamp", columns="city_name", values="temperature")',
            '    paris = df[df["city_name"] == "Paris"].set_index("timestamp")',
            '    X = temp.add_prefix("t_").join((temp - temp.shift(24)).add_prefix("d24_"))',
            "    for lag in (1, 3, 6, 12, 18, 24, 48):",
            '        X[f"paris_lag{lag}"] = paris["temperature"].shift(lag)',
            '    for col in ("humidity", "clouds", "wind_speed", "rain"):',
            '        X[f"paris_{col}"] = paris[col]',
            '    wind = np.radians(paris["wind_direction"])',
            '    X["wind_sin"], X["wind_cos"] = np.sin(wind), np.cos(wind)',
            "    hour, day = X.index.hour, X.index.dayofyear",
            '    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)',
            '    X["day_sin"], X["day_cos"] = np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25)',
            "    return X",
            "",
            "",
            "class Agent:",
            "    def __init__(self):",
            '        with open(os.path.join(os.path.dirname(__file__), "model.json")) as f:',
            "            self.model = json.load(f)",
            "",
            "    def predict(self, rows):",
            "        x = features(pd.DataFrame(rows)).iloc[-1]",
            "        out = {}",
            '        for key, m in self.model.items():          # "t+6h", "t+48h"',
            '            center = m["intercept"] + np.dot(x[m["columns"]], m["coef"])',
            '            out[key] = (center + np.array(m["residuals"])).tolist()',
            "        return out",
        ),
        md(
            "## 3. Train",
            "",
            "Fit on 2020–2023, measure the errors on 2024 (their 50 quantiles become",
            "the ensemble), keep 2025–2026 for the local score.",
        ),
        code(
            "from sklearn.linear_model import Ridge",
            "from agent import features",
            "",
            "X = features(df)",
            'paris = X["t_Paris"]',
            "levels = (np.arange(50) + 0.5) / 50",
            "model = {}",
            'for h, key in [(6, "t+6h"), (48, "t+48h")]:',
            "    y = paris.shift(-h)                      # the temperature h hours later",
            "    ok = X.notna().all(axis=1) & y.notna()",
            '    fit = ok & (X.index < "2024-01-01")',
            '    cal = ok & (X.index >= "2024-01-01") & (X.index < "2025-01-01")',
            "    reg = Ridge(alpha=1.0).fit(X[fit], y[fit])",
            "    errors = y[cal] - reg.predict(X[cal])",
            '    model[key] = {"columns": list(X.columns), "coef": reg.coef_.tolist(),',
            '                  "intercept": float(reg.intercept_),',
            '                  "residuals": np.quantile(errors, levels).tolist()}',
            '    print(f"{key}: mean absolute error on 2024 = {errors.abs().mean():.2f} °C")',
            "",
            'with open("model.json", "w") as f:',
            "    json.dump(model, f)",
        ),
        md(
            "## 4. Score it like the challenge",
            "",
            "One forecast every 6 hours from 2025-01 to the end of the data (the last",
            "months are the live feed, the provider the challenge scores on), with the CRPS:",
            "`mean|x - y| - ½ mean|x_i - x_j|`, in °C, lower is better. The",
            "persistence line (the last value, one member) is the bar to clear.",
        ),
        code(
            "from agent import Agent",
            "",
            "def crps(x, y):",
            "    x = np.asarray(x)",
            "    return np.abs(x - y).mean() - 0.5 * np.abs(x[:, None] - x[None, :]).mean()",
            "",
            'hours = pd.to_datetime(df["timestamp"].iloc[::N_CITIES]).reset_index(drop=True)',
            'temp_paris = df.loc[df["city_name"] == "Paris", "temperature"].to_numpy()',
            'start = hours.searchsorted(pd.Timestamp("2025-01-01 05:00", tz="UTC"))',
            "agent, scores = Agent(), []",
            "for k in range(start, len(hours) - 48, 6):",
            "    out = agent.predict(rows_at(k))",
            '    model_crps = (crps(out["t+6h"], temp_paris[k + 6]) + crps(out["t+48h"], temp_paris[k + 48])) / 2',
            "    last = temp_paris[k]",
            "    persistence = (abs(last - temp_paris[k + 6]) + abs(last - temp_paris[k + 48])) / 2",
            "    scores.append((model_crps, persistence))",
            "",
            'model_score, persistence_score = np.mean(scores, axis=0)',
            'print(f"{len(scores)} forecasts — model {model_score:.3f} °C, persistence {persistence_score:.3f} °C")',
        ),
        md(
            "## 5. Submit",
            "",
            "`agent.py` and `model.json` go up together; runner 181 (numpy + pandas)",
            "is enough. The first score arrives about 48 h after the deploy.",
        ),
        code(
            "import mlarena",
            'if "google.colab" in sys.modules:',
            "    from google.colab import userdata",
            '    os.environ["MLARENA_API_KEY"] = userdata.get("MLARENA_API_KEY")',
            'client = mlarena.connect(api_key=os.environ["MLARENA_API_KEY"])',
            f"client.submit({CHALLENGE_ID}, files=[\"agent.py\", \"model.json\"],",
            '              submission_name="ridge", runtime_id=181, wait=True)',
        ),
    ]
    return nb.notebook(cells)


def main() -> None:
    nb._COUNTER = itertools.count()     # byte-identical on every regeneration
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / NAME
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(REPO)}\n      {COLAB}")


if __name__ == "__main__":
    main()
