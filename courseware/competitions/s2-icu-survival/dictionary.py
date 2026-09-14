"""Renders DICTIONARY.md: every served column, its meaning, unit, fill rate,
distinct count and example values.

Meanings come from the SUPPORT documentation (Harrell, hbiostat.org/data/repo/
supportdesc; Knaus et al., Ann Intern Med 1995) and from the charting
conventions of the five sites of this export. The fill rates, distinct counts
and examples are computed from the shipped `train.csv.gz`, so they describe
exactly the file a student downloads: dirty spellings, text tokens and
placeholders included.

`SERVED` is the list of feature columns, in the shipped order, without the
`id` and the target. `python dictionary.py` renders `DICTIONARY.md` next to
this file from `data/train.csv.gz`, and fails if that file does not exist.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ID_COLUMN = "id"
TARGET = "dead"

_FLAG = "0/1"
_TEXT = "—"

# --------------------------------------------------------------------------- #
# the served columns, in the shipped order (id first, target last, not listed)
# --------------------------------------------------------------------------- #
SERVED: list[str] = [
    "site", "age", "sex", "race", "edu", "income",
    "dzgroup", "dzclass", "num_co", "ca", "diabetes", "dementia", "hday",
    "scoma", "meanbp", "hrt", "resp", "temp",
    "wblc", "pafi", "alb", "bili", "crea", "sod", "ph", "glucose", "bun", "urine",
    "adlp", "adls",
    "charges",
]

GROUPS: list[tuple[str, list[str]]] = [
    ("Identification & site", ["site", "hday"]),
    ("Demographics & social", ["age", "sex", "race", "edu", "income"]),
    ("Diagnosis & comorbidity", ["dzgroup", "dzclass", "num_co", "ca", "diabetes", "dementia"]),
    ("Vital signs (day 3)", ["scoma", "meanbp", "hrt", "resp", "temp"]),
    ("Laboratory (day 3)", ["wblc", "pafi", "alb", "bili", "crea", "sod", "ph",
                            "glucose", "bun", "urine"]),
    ("Functional status", ["adlp", "adls"]),
    ("Billing", ["charges"]),
]

MEANINGS: dict[str, tuple[str, str]] = {
    "site": ("Hospital site, A to E. SUPPORT ran in five US academic medical centres; "
             "each site exported its chart in its own units (see `crea`, `temp`).", "code"),
    "age": ("Age at study entry. Some sites record 999 when the age is unknown.", "years"),
    "sex": ("Sex of the patient. Free-text entry: the spellings differ by site and by "
            "clerk (full word or initial, upper or lower case).", _TEXT),
    "race": ("Race as recorded at admission: white, black, hispanic, asian, other. "
             "Free-text entry: case and trailing spaces vary. Site E writes "
             "'unknown' where the other sites leave the cell empty.", _TEXT),
    "edu": ("Years of education. Empty when the social interview did not take place.", "years"),
    "income": ("Household income bracket, from the interview: under $11k, $11-$25k, "
               "$25-$50k, >$50k. Empty when the patient could not be asked or declined.", _TEXT),
    "dzgroup": ("Qualifying diagnosis, one of eight groups: ARF/MOSF w/Sepsis, CHF, COPD, "
                "Cirrhosis, Colon Cancer, Coma, Lung Cancer, MOSF w/Malig "
                "(MOSF = multiple organ system failure; ARF = acute respiratory failure).", _TEXT),
    "dzclass": ("Disease class: ARF/MOSF, COPD/CHF/Cirrhosis, Cancer, Coma. A grouping "
                "of `dzgroup` into four.", _TEXT),
    "num_co": ("Number of comorbid conditions recorded at entry, 0 to 9.", "count"),
    "ca": ("Cancer status: no, yes (non-metastatic), metastatic.", _TEXT),
    "diabetes": ("1 if diabetes is among the comorbidities.", _FLAG),
    "dementia": ("1 if dementia is among the comorbidities.", _FLAG),
    "hday": ("Hospital day on which the patient entered the study (1 = the day of "
             "admission).", "days"),
    "scoma": ("SUPPORT coma score on day 3, derived from the Glasgow Coma Scale: "
              "0 = fully conscious, 100 = deep coma.", "0-100"),
    "meanbp": ("Mean arterial blood pressure on day 3 (worst value of the day).", "mmHg"),
    "hrt": ("Heart rate on day 3 (worst value of the day).", "/min"),
    "resp": ("Respiratory rate on day 3 (worst value of the day).", "/min"),
    "temp": ("Body temperature on day 3 (worst value of the day). Charted in °C at "
             "sites A, B, D and E and in °F at site C. Some rows record 0.0 when the "
             "temperature was not taken.", "°C or °F by site"),
    "wblc": ("White blood cell count on day 3.", "×10⁹/L (thousands/µL)"),
    "pafi": ("PaO2/FiO2 ratio on day 3, from the arterial blood gas. Empty when no "
             "blood gas was drawn.", "mmHg"),
    "alb": ("Serum albumin on day 3. Empty when the liver panel was not ordered.", "g/dL"),
    "bili": ("Serum bilirubin on day 3. Empty when the liver panel was not ordered.", "mg/dL"),
    "crea": ("Serum creatinine on day 3. Reported in mg/dL at sites A, B and C and in "
             "µmol/L at sites D and E (1 mg/dL = 88.4 µmol/L).", "mg/dL or µmol/L by site"),
    "sod": ("Serum sodium on day 3. A few rows carry the value with the decimal point "
            "dropped (ten times too large).", "mEq/L"),
    "ph": ("Arterial pH on day 3, from the arterial blood gas. Empty when no blood gas "
           "was drawn.", "—"),
    "glucose": ("Serum glucose on day 3. Empty when the chemistry panel was not "
                "ordered, except at site B, which writes the text 'not done' instead; "
                "pandas therefore reads the whole column as text.", "mg/dL"),
    "bun": ("Blood urea nitrogen on day 3. Empty when the chemistry panel was not "
            "ordered.", "mg/dL"),
    "urine": ("Urine output on day 3, 24-hour collection. Empty when not collected; "
              "0 is a measured value (anuria).", "mL/24h"),
    "adlp": ("Activities of daily living the patient reported being unable to do on "
             "day 3, 0 to 7. Empty when the patient could not be interviewed.", "count 0-7"),
    "adls": ("The same count reported by the surrogate (family) on day 3, 0 to 7. "
             "Empty when no surrogate was interviewed.", "count 0-7"),
    "charges": ("Total hospital charges for the stay, billed at discharge; empty for "
                "patients not yet discharged.", "USD"),
}

_HEADER = """# DICTIONARY — the columns of `train.csv.gz` and `test.csv.gz`

Both files carry `id` and the {n} columns below, in this order; `train.csv.gz`
adds the target `dead` (1 if the patient died within 60 days of study entry,
else 0).

The values are **as a hospital export would look**: units that differ by
site, free-text spellings, placeholders, a text token in a numeric column,
labs that are empty because nobody ordered them. Nothing has been cleaned or
recoded. Deciding what each column needs is the exercise; `EXPERTISE.md` gives
the reasons.

- **Meaning:** from the SUPPORT documentation
  ([hbiostat.org/data/repo/supportdesc](https://hbiostat.org/data/repo/supportdesc))
  and the charting conventions of the five sites.
- **Read as:** the dtype `pd.read_csv` gives the column. `number` columns can
  still be codes or flags; `text` columns can still hold an order, or numbers.
- **Filled:** share of the {rows:,} rows of `train.csv.gz` with a value.
- **Distinct:** number of distinct values among the filled rows.
- **Examples:** for text, the distinct spellings actually present (all of them
  when there are few, else the most frequent) with their share of the filled
  rows; for numbers, the minimum, median and maximum.
"""


def _fmt_num(x) -> str:
    x = float(x)
    if x.is_integer():
        return f"{int(x):,}"
    return f"{x:,.1f}" if abs(x) >= 1000 else f"{x:g}"


def _cell(value: str) -> str:
    value = value if len(value) <= 40 else value[:37] + "…"
    return "`" + value.replace("|", "\\|") + "`"


def _examples(s: pd.Series) -> str:
    filled = s.dropna()
    if filled.empty:
        return "*(always empty)*"
    if pd.api.types.is_numeric_dtype(s):
        return (f"min {_fmt_num(filled.min())} · median {_fmt_num(filled.median())} · "
                f"max {_fmt_num(filled.max())}")
    text = filled.astype(str)
    as_num = pd.to_numeric(text, errors="coerce")
    if as_num.notna().mean() > 0.5:
        # a numeric column that a few text tokens turned into text: show the
        # numeric part as numbers and every token that is not a number
        tokens = text[as_num.isna()].value_counts(normalize=False)
        parts = [f"min {_fmt_num(as_num.min())} · median {_fmt_num(as_num.median())} · "
                 f"max {_fmt_num(as_num.max())}"]
        parts += [f"{_cell(tok)} ({n / len(text):.1%})" for tok, n in tokens.items()]
        return " · ".join(parts)
    shares = text.value_counts(normalize=True)
    shown = shares if len(shares) <= 8 else shares.head(4)
    parts = [f"{_cell(value)} ({share:.0%})" for value, share in shown.items()]
    more = len(shares) - len(shown)
    return " · ".join(parts) + (f" · *+{more} more*" if more > 0 else "")


def render(train: pd.DataFrame, path: Path) -> None:
    """Write DICTIONARY.md for the shipped `train` frame at `path`. Fail fast on
    a column list that drifted from MEANINGS / GROUPS or from the frame."""
    grouped = [c for _, cols in GROUPS for c in cols]
    if sorted(grouped) != sorted(SERVED) or len(grouped) != len(SERVED):
        raise SystemExit(f"GROUPS out of sync with SERVED: {sorted(set(grouped) ^ set(SERVED))}")
    if list(MEANINGS) != SERVED:
        raise SystemExit("MEANINGS must list exactly SERVED, in order")
    expected = [ID_COLUMN, *SERVED, TARGET]
    if list(train.columns) != expected:
        raise SystemExit(f"train columns {list(train.columns)} != {expected}")

    lines = [_HEADER.format(n=len(SERVED), rows=len(train))]
    for group, cols in GROUPS:
        lines.append(f"\n## {group}\n")
        lines.append("| column | meaning | unit | read as | filled | distinct | examples |")
        lines.append("|---|---|---|---|---|---|---|")
        for col in cols:
            meaning, unit = MEANINGS[col]
            s = train[col]
            kind = "number" if pd.api.types.is_numeric_dtype(s) else "text"
            lines.append(f"| `{col}` | {meaning} | {unit} | {kind} | "
                         f"{s.notna().mean():.1%} | {s.nunique():,} | {_examples(s)} |")
    lines.append("\n## Target (`train.csv.gz` only)\n")
    lines.append("| column | meaning | unit | read as | filled | distinct | examples |")
    lines.append("|---|---|---|---|---|---|---|")
    y = train[TARGET]
    lines.append(f"| `{TARGET}` | 1 if the patient died within 60 days (2 months) of study "
                 f"entry, 0 if alive at 60 days. | {_FLAG} | number | {y.notna().mean():.1%} | "
                 f"{y.nunique()} | {y.mean():.1%} are 1 |")
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    src = here / "data" / "train.csv.gz"
    if not src.is_file():
        raise SystemExit(f"{src} does not exist: run prepare_data.py first")
    out = here / "DICTIONARY.md"
    render(pd.read_csv(src), out)
    print(f"wrote {out}")
