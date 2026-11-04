"""Paris Temperature Forecast — the data tour (Colab), for teachers.

A short, presentable walk through the data behind challenge 203: the 45 cities,
Paris's daily and seasonal cycles, how predictable +6 h and +48 h are, why the
other cities help, exactly what an agent receives, how the CRPS rewards an
honest spread, the baselines, and the two weather providers. No API key, no
submission: it only reads the two public CSVs.

Writes one file; safe to run:

    uv run python courseware/tools/notebooks/paris_temperature_data.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)
import paris_temperature as agent_nb  # noqa: E402  (same folder, ids and URLs)

REPO = agent_nb.REPO
OUT = agent_nb.OUT
NAME = "paris-temperature-data-tour.ipynb"
BASE = f"https://raw.githubusercontent.com/{agent_nb.GITHUB}/main/{OUT.relative_to(REPO)}/"
COLAB = (f"https://colab.research.google.com/github/{agent_nb.GITHUB}/blob/main/"
         f"{OUT.relative_to(REPO)}/{NAME}")
CHALLENGE_ID = agent_nb.CHALLENGE_ID
ARCHIVE = "weather_europe_2020_2026.csv.gz"
LIVE = agent_nb.LIVE

md, code = nb.md, nb.code


def build() -> dict:
    cells = [
        md(
            "# Paris Temperature Forecast — a tour of the data",
            "",
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})",
            "",
            f"**The challenge** ([ML-Arena #{CHALLENGE_ID}](https://ml-arena.com/viewchallenge/{CHALLENGE_ID})):",
            "every 6 hours, an agent receives the last 72 hours of weather in 45 European",
            "cities and returns plausible Paris temperatures **6 h and 48 h later**. Each",
            "forecast is scored 48 h later, once the temperature is known, with the **CRPS**",
            "(°C, lower is better).",
            "",
            "This notebook needs no account: it reads two public files and draws every",
            "figure in about a minute. The student starter is",
            f"[a separate notebook]({agent_nb.COLAB}).",
        ),
        code(
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            'plt.rcParams.update({"figure.figsize": (10, 4), "axes.grid": True, "grid.alpha": 0.3})',
            f'BASE = "{BASE}"',
            f'archive = pd.read_csv(BASE + "{ARCHIVE}")      # Open-Meteo reanalysis, 2020-01 → 2026-03',
            f'live = pd.read_csv(BASE + "{LIVE}")   # the live feed since',
            "df = pd.concat([archive, live], ignore_index=True)",
            'df["time"] = pd.to_datetime(df["timestamp"])',
            'print(f"{len(df):,} rows = {df.time.nunique():,} hours x {df.city_name.nunique()} cities, "',
            '      f"{df.time.min():%Y-%m-%d} → {df.time.max():%Y-%m-%d}")',
            'df.drop(columns="time").tail(3)',
        ),
        md(
            "## 1. One row per city and hour",
            "",
            "13 columns, hourly, UTC, no missing hour. `temperature` (°C) of Paris is the",
            "target; everything else is a possible input. `visibility` is always empty.",
            "",
            "| column | unit |",
            "|---|---|",
            "| `temperature` | °C |",
            "| `rain`, `snow` | mm in the hour |",
            "| `wind_speed`, `wind_direction` | km/h, degrees the wind comes **from** |",
            "| `humidity`, `clouds` | % |",
        ),
        code(
            'cities = df.drop_duplicates("city_name")[["city_name", "latitude", "longitude"]]',
            "fig, ax = plt.subplots(figsize=(7, 6))",
            'ax.scatter(cities.longitude, cities.latitude, s=25, color="tab:blue")',
            'paris = cities[cities.city_name == "Paris"]',
            'ax.scatter(paris.longitude, paris.latitude, s=160, color="tab:red", marker="*", zorder=3)',
            "for _, c in cities.iterrows():",
            "    ax.annotate(c.city_name, (c.longitude, c.latitude), fontsize=7, xytext=(3, 3),",
            '                textcoords="offset points")',
            'ax.set(xlabel="longitude", ylabel="latitude", title="The 45 cities (Paris in red)")',
            'ax.set_aspect(1 / np.cos(np.radians(48)))',
            "plt.show()",
        ),
        md(
            "## 2. Paris: a seasonal cycle and a daily cycle",
            "",
            "Most of the variance is calendar: the season sets the level, the hour of the",
            "day adds a ±4 °C swing. What is left — the weather — is what the forecast is",
            "about.",
        ),
        code(
            'p = df[df.city_name == "Paris"].set_index("time")["temperature"]',
            "",
            "fig, ax = plt.subplots()",
            'p.resample("D").mean().plot(ax=ax, lw=0.6, color="tab:gray", label="daily mean")',
            'p.resample("D").mean().rolling(30, center=True).mean().plot(ax=ax, color="tab:red", label="30-day mean")',
            f'ax.axvline(pd.Timestamp(live.timestamp.iloc[0]), color="k", ls="--", lw=1)',
            'ax.set(ylabel="°C", xlabel="", title="Paris temperature (dashed: the live feed starts)")',
            "ax.legend()",
            "plt.show()",
            "",
            "cycle = p.groupby([p.index.month, p.index.hour]).mean().unstack()",
            "fig, ax = plt.subplots(figsize=(10, 3.5))",
            'im = ax.imshow(cycle, aspect="auto", cmap="coolwarm", origin="lower",',
            "               extent=(-0.5, 23.5, 0.5, 12.5))",
            'ax.set(xlabel="hour of day (UTC)", ylabel="month", title="Mean Paris temperature, month x hour")',
            'fig.colorbar(im, label="°C")',
            "plt.show()",
        ),
        md(
            "## 3. How hard are +6 h and +48 h?",
            "",
            "Three naive forecasts, and their mean absolute error over the whole history:",
            "",
            "- **last value**: the temperature now;",
            "- **same hour, last day**: the last observed temperature at the target's hour",
            "  of day (the starter agent's guess);",
            "- **climatology**: the mean for that month and hour.",
        ),
        code(
            "clim = p.groupby([p.index.month, p.index.hour]).transform(\"mean\")",
            "rows = []",
            "for h in (6, 48):",
            "    target = p.shift(-h)",
            "    same_hour = p.shift(24 - h) if h < 24 else p",
            "    rows.append({",
            '        "lead": f"+{h} h",',
            '        "last value": (target - p).abs().mean(),',
            '        "same hour, last day": (target - same_hour).abs().mean(),',
            '        "climatology": (target - clim.shift(-h)).abs().mean(),',
            "    })",
            'pd.DataFrame(rows).set_index("lead").round(2)',
        ),
        md(
            "At +6 h the daily cycle dominates, so \"same hour yesterday\" beats \"now\".",
            "At +48 h every naive forecast is ~2–3 °C off: the weather has moved on, and",
            "the information has to come from somewhere else.",
            "",
            "## 4. The other 44 cities: the weather comes from the west",
            "",
            "Correlate the change of Paris's temperature over the next 24 h with each",
            "city's change over the **last** 24 h. Cities upwind (west) announce what",
            "reaches Paris; cities downwind (east) do not.",
        ),
        code(
            'wide = df.pivot(index="time", columns="city_name", values="temperature")',
            "past_change = wide - wide.shift(24)",
            'paris_next = wide["Paris"].shift(-24) - wide["Paris"]',
            "corr = past_change.corrwith(paris_next).drop(\"Paris\")",
            'lon = cities.set_index("city_name").longitude.drop("Paris")',
            "",
            "fig, ax = plt.subplots(figsize=(9, 4))",
            'ax.scatter(lon, corr[lon.index], color="tab:blue")',
            "for name in corr.abs().sort_values().index[-8:]:",
            "    ax.annotate(name, (lon[name], corr[name]), fontsize=8, xytext=(3, 3), textcoords=\"offset points\")",
            'ax.axvline(cities.set_index("city_name").longitude["Paris"], color="tab:red", ls="--")',
            'ax.set(xlabel="longitude (Paris dashed)", ylabel="correlation",',
            '       title="Last-24 h change in a city vs next-24 h change in Paris")',
            "plt.show()",
        ),
        md(
            "## 5. What an agent receives",
            "",
            "`predict(rows)` gets the last 72 hours × 45 cities — rows of exactly this",
            "file. `pd.DataFrame(rows)` is the same frame as `pd.read_csv` for those hours,",
            "so whatever a student computes on the history runs unchanged in the agent.",
        ),
        code(
            'window = df.drop(columns="time").tail(72 * 45)',
            "rows = window.to_dict(\"records\")          # what predict() receives",
            "print(len(rows), \"rows; last timestamp:\", rows[-1][\"timestamp\"])",
            "pd.DataFrame(rows).head(3)",
        ),
        md(
            "The answer is a dict of plausible temperatures for 6 h and 48 h after the last",
            "timestamp — 1 to 100 values each:",
            "",
            "```python",
            'return {"t+6h": [11.8, 12.4, 13.1], "t+48h": [9.0, 10.5, 12.2]}',
            "```",
            "",
            "## 6. The score: CRPS rewards an honest spread",
            "",
            "`CRPS = mean|x_i − y| − ½ mean|x_i − x_j|`. With one value it is the absolute",
            "error. With many, the first term is the average miss and the second refunds the",
            "spread: a forecast that is uncertain *and says so* scores better than a",
            "confident one that misses.",
        ),
        code(
            "def crps(x, y):",
            "    x = np.asarray(x, dtype=float)",
            "    return np.abs(x - y).mean() - 0.5 * np.abs(x[:, None] - x[None, :]).mean()",
            "",
            "y = 14.0                                    # what happened",
            "z = np.random.default_rng(0).standard_normal(50)",
            "forecasts = {",
            '    "point forecast, 2 °C off": [12.0],',
            '    "narrow ensemble, 2 °C off": 12.0 + 0.3 * z,',
            '    "honest ensemble, 2 °C off": 12.0 + 2.0 * z,',
            '    "far too wide": 12.0 + 8.0 * z,',
            "}",
            "fig, ax = plt.subplots(figsize=(9, 3))",
            "for i, (name, x) in enumerate(forecasts.items()):",
            '    ax.scatter(x, np.full(len(x), i), alpha=0.4, s=15)',
            '    ax.text(25, i, f"CRPS {crps(x, y):.2f}", va="center")',
            'ax.axvline(y, color="tab:green", lw=2, label="observed")',
            'ax.set(yticks=range(len(forecasts)), yticklabels=list(forecasts), xlim=(-8, 30), xlabel="°C")',
            "ax.legend(loc=\"lower left\")",
            "plt.show()",
        ),
        md(
            "## 7. Baselines",
            "",
            "Replayed through the challenge's own scoring on 1,700 forecasts (2025-01 →",
            "2026-03), fitted on 2020–2024 only:",
            "",
            "| Agent | Score (°C) | CRPS +6 h | CRPS +48 h |",
            "|---|---|---|---|",
            "| Last observed value, one member | 2.986 | 3.061 | 2.911 |",
            "| Paris month × hour climatology | 2.189 | 2.189 | 2.189 |",
            "| **Starter agent** (same hour, last day + normal spread) | **1.821** | 1.554 | 2.087 |",
            "| Ridge on 45-city features (the student notebook) | 1.354 | 0.981 | 1.727 |",
            "| Gradient boosting, same features | 1.262 | 0.812 | 1.713 |",
            "",
            "The starter, one week of forecasts against what happened:",
        ),
        code(
            'week = p["2025-11-10":"2025-11-17"]',
            "centre = p.shift(24)[week.index]          # +6 h guess: the same hour one day earlier",
            "spread = 2.8                              # °C, the starter's normal spread at +6 h",
            "fig, ax = plt.subplots()",
            "ax.fill_between(week.index, centre - 1.645 * spread, centre + 1.645 * spread,",
            '                color="tab:blue", alpha=0.15, lw=0, label="90 % of its ensemble")',
            'ax.plot(week.index, centre, color="tab:blue", label="centre of its ensemble")',
            'ax.plot(week.index, week, color="k", label="observed")',
            'ax.set(ylabel="°C", title="Starter agent at +6 h: same hour yesterday, with its spread")',
            "ax.legend()",
            "plt.show()",
        ),
        md(
            "## 8. Two providers",
            "",
            "The archive is the **Open-Meteo** reanalysis. The live feed — what agents",
            "receive and what they are scored against — is **OpenWeatherMap** (≈3 % of hours",
            "filled from Open-Meteo), and the second file holds it since the archive ends.",
            "Where both exist for the same city and hour they differ by ~0.9 °C (median),",
            "so a model tuned only on the archive scores worse live than in a replay: the",
            "student notebook's Ridge scores 1.354 °C over 2025-01 → 2026-03 (archive only)",
            "and 1.443 °C once the eight live months are included. Training on the recent",
            "months too, or at least validating on them, is the remedy.",
        ),
        code(
            'start = pd.Timestamp(live.timestamp.iloc[0])',
            "fig, ax = plt.subplots()",
            'p.resample("MS").mean().plot(ax=ax, marker="o", ms=3, color="tab:gray", label="monthly mean")',
            'ax.axvspan(start, p.index.max(), color="tab:orange", alpha=0.15, label="live feed (OpenWeatherMap)")',
            'ax.set(ylabel="°C", xlabel="", title="Paris, monthly mean temperature (before the band: Open-Meteo archive)")',
            "ax.legend()",
            "plt.show()",
        ),
        md(
            "## 9. Preparing students",
            "",
            "A progression that fits a lab session or two:",
            "",
            "1. **Read the frame**: pivot to one column per city, plot Paris, find the",
            "   daily and seasonal cycles (sections 1–2).",
            "2. **Point forecast**: persistence, then same-hour-yesterday; measure the MAE",
            "   at +6 h and +48 h (section 3).",
            "3. **Features**: lags of Paris, other cities now and their 24 h change, hour",
            "   and season; a linear model per lead time (section 4, the student notebook).",
            "4. **From a point to a distribution**: add the quantiles of the model's errors",
            "   on a held-out year — the CRPS drops (section 6).",
            "5. **Stronger models**: gradient boosting, quantile regression, recent data.",
            "",
            "Things to tell them up front:",
            "",
            "- the first score appears **~48 h after deploying** (8 runs without a score is",
            "  normal);",
            "- timestamps are **UTC**; \"+6 h\" is after the **last** timestamp received;",
            "- validate on a later period than you train on — the challenge is always the",
            "  future.",
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
