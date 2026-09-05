# Files and Formats

A file format is a set of promises. Knowing which promises a format makes — and
which it does not — is the difference between a clean load and a week of silent
type coercion.

<!-- notes: 30 minutes. The CSV-has-no-types point is the one to hammer. Live
demo: read a CSV with a leading-zero postcode column and watch it become an int.
-->

---

## CSV — the lowest common denominator

```python
df = pd.read_csv("data.csv")
```

Text, one line per row, a delimiter between fields. Universally readable, and it
carries **no type information at all**.

Everything is a string on disk; pandas guesses. The guess is wrong for postcodes,
phone numbers, IDs with leading zeros, and any column that is 99% integers and 1%
`"N/A"`.

---

## Reading a CSV properly

```python
df = pd.read_csv(
    "data.csv",
    dtype={"zipcode": str, "user_id": str},
    parse_dates=["signup_date"],
    na_values=["", "NA", "N/A", "-", "null"],
)
```

Four arguments that prevent four different silent disasters:

- `dtype` — stops the leading zero from being eaten
- `parse_dates` — stops dates staying strings
- `na_values` — the source's idea of "missing" is rarely `NaN`
- also worth setting: `sep`, `encoding` for non-UTF-8 exports

---

## The encoding problem

```python
df = pd.read_csv("export.csv", encoding="latin1")
```

A file exported from Excel on a French Windows machine is very often `cp1252`,
not UTF-8. The symptom is `UnicodeDecodeError`, or worse, `Ã©` where `é` should
be.

Do not "fix" this with `errors="ignore"`. That deletes the characters and you
lose the information silently — exactly the failure mode this course argues
against.

---

## Excel — data with formatting attached

```python
df = pd.read_excel("data.xlsx", sheet_name="2025", header=2)
```

Multiple sheets, merged cells, a title in row 1, a footnote in the last row,
formulas that evaluate to `#REF!`. It is a *document*, not a dataset.

Load it once, assert what you expect, and write it out as Parquet. Do not read
the `.xlsx` again in your pipeline.

---

## Parquet — the one you should default to

```python
df.to_parquet("data.parquet")
df = pd.read_parquet("data.parquet", columns=["user_id", "amount"])
```

Columnar, binary, compressed, and it **stores the schema**. A `datetime` written
is a `datetime` read.

| | CSV | Parquet |
|---|---|---|
| Types preserved | no | yes |
| Read one column | reads all | reads one |
| Size (typical) | 1× | 0.2–0.4× |
| Human-readable | yes | no |
| Appendable by hand | yes | no |

---

## The rule

> CSV to exchange with humans. Parquet for everything your pipeline touches.

The first thing your ingestion step should do is convert. From then on, reads are
five times faster and the schema stops being a guess.

---

## JSON — nested and self-describing

```json
{"user": {"id": 7, "tags": ["a", "b"]}, "amount": 12.5}
```

```python
df = pd.read_json("data.json")                       # a JSON array
df = pd.read_json("events.jsonl", lines=True)        # one object per line
```

JSON Lines (`.jsonl`) is the format APIs and log pipelines actually emit, and the
only one of the two you can stream without loading the whole file.

---

## Flattening nested JSON

```python
import json
records = [json.loads(line) for line in open("events.jsonl")]
df = pd.json_normalize(records, sep="_")
```

`json_normalize` turns `{"user": {"id": 7}}` into a `user_id` column.

It does **not** solve lists. A field holding `["a", "b"]` needs an explicit
decision: explode into multiple rows, or encode as a set of indicator columns.
That decision belongs to you, not to a default.

---

## XML

```python
df = pd.read_xml("data.xml", xpath="//employee")
```

Still the native format of institutional and government data feeds. The `xpath`
argument is mandatory in practice — it selects the repeating element that
becomes a row.

---

## Text and images

```python
text = open("doc.txt", encoding="utf-8").read()
```

```python
from PIL import Image
img = Image.open("photo.jpg")
print(img.size, img.mode)          # (1920, 1080) RGB
```

For unstructured data the "loading" step is trivial and the *representation* step
is the entire problem — Sessions 5 to 8.

Keep a manifest: a CSV of `path, label, split, source`, with the media on disk
next to it. Never put images in a dataframe cell.

---

## Choosing

| Situation | Format |
|---|---|
| Handing data to a non-programmer | CSV |
| Anything inside your pipeline | Parquet |
| API payloads, logs, streams | JSON Lines |
| Institutional / government feeds | XML |
| Someone sent a spreadsheet | XLSX → convert immediately |
| Media | files on disk + a manifest |

---

## Verify at the boundary

```python
df = pd.read_parquet("data.parquet")
assert df["user_id"].dtype == object
assert df["ts"].is_monotonic_increasing
assert df["amount"].notna().all()
```

Three assertions at load time cost thirty seconds to write and save the
afternoon you would otherwise spend explaining a negative revenue figure.

This is the fail-fast principle applied to data: crash at the boundary, not deep
in a training loop at epoch 40.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import io, pandas as pd

   csv = "zipcode,city\n07001,Bobigny\n"
   print(pd.read_csv(io.StringIO(csv))["zipcode"][0])                       # -> 7001
   print(pd.read_csv(io.StringIO(csv),
                     dtype={"zipcode": str})["zipcode"][0])                 # -> 07001
   ```

   **Answer.** CSV carries no type information, so pandas guessed `int` and ate
   the leading zero. `dtype=` is the argument that prevents it.

2. You write a dataframe with a `datetime` column to CSV and to Parquet, and read
   both back. Which one gives you a `datetime` again, and what is the rule the
   lesson draws from that?

   **Answer.** Parquet — it stores the schema, so a `datetime` written is a
   `datetime` read; the CSV round-trip returns a string. The rule: CSV to
   exchange with humans, Parquet for everything your pipeline touches.

3. A CSV exported from Excel raises `UnicodeDecodeError`. Why is
   `errors="ignore"` the wrong repair?

   **Answer.** It deletes the characters it cannot decode, so the information is
   lost silently. Find the real encoding instead — a French Windows export is
   very often `cp1252`/`latin1`, not UTF-8.
