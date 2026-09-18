# Files and Formats

A file format is a set of promises. Knowing which promises a format makes — and
which it does not — is the difference between a clean load and a week of silent
type coercion.

<!-- notes: 30 minutes. The CSV-has-no-types point is the one to hammer. Live
demo: read a CSV with a leading-zero postcode column and watch it become an int.
-->

---

## Look at the file before you load it

A file's metadata tells you how to read it:

- **format and extension** — `.csv`, `.xlsx`, `.parquet`, `.json`, `.xml`
- **encoding** — UTF-8, or a legacy one such as `cp1252`
- **structure** — is there a header row, which delimiter, any title lines?
- **size and modification date** — in memory or not, and how fresh

```python
from pathlib import Path

p = Path("data.special_format")
with p.open("rb") as f:          # rb: bytes "r" : text
    print(f.read(120))           # delimiter, header, BOM, line endings
```

Thirty seconds of reading raw bytes answers the questions `read_csv` would
otherwise answer by guessing.

---

## CSV — the lowest common denominator

```text
id,name,age,city
1,John Doe,30,New York
2,Jane Smith,25,Los Angeles
```

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

## Delimiters and headers

"Comma-separated" is a convention, not a guarantee. Where the comma is the
decimal separator, as in French exports, the delimiter becomes `;`:

```python
df = pd.read_csv("store_b.csv", sep=";", decimal=",", header=1)
```

- `header=1` — the column names are on line 2, under a title line
- `header=None, names=[...]` — the file has no header row at all
- `Unnamed: 10` — a trailing delimiter on every line created an empty column

Drop that empty column by name, and check that it really is empty first.

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

![encoding.png](assets/collect/encoding.png)

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

## A workbook holds several tables

```python
sheets = pd.read_excel("store_c.xlsx", sheet_name=None)
print({name: s.shape for name, s in sheets.items()})
```

`sheet_name=None` returns a dict of every sheet, keyed by name — the only way to
see what the workbook contains before choosing. Two sheets describing the same
products have to be *joined*, on a key that may be spelled differently in each.

Cells hold text, numbers, dates or formulas, and `usecols="A:C"` restricts the
read to a column range. `pandas` needs the `openpyxl` package for `.xlsx`.

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

## Parquet: compression and partitions

```python
df.to_parquet("sales/", partition_cols=["year"])   # sales/year=2025/...
df = pd.read_parquet("sales/", filters=[("year", "==", 2025)])
```

Compression is per column and on by default (`snappy`); `compression="gzip"`
writes smaller files that are slower to read.

Partitioning writes one directory per value of a column. A read that filters on
that column skips the other directories entirely — the same "do not read what
you do not need" principle, applied to whole files. The partition column comes
back as a `category`.

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

JSON has six value types — string, number, boolean, null, object, array — and
**no date type**. Every timestamp arrives as a string and has to be parsed.

---

## Flattening nested JSON

```python
import json
records = [json.loads(line) for line in open("events.jsonl")]
df = pd.json_normalize(records, sep="_")
```

`json_normalize` turns `{"user": {"id": 7}}` into a `user_id` column. When the
records sit inside a wrapper, as in `{"employees": [...]}`, pass
`record_path="employees"` to reach them.

It does **not** solve lists. A field holding `["a", "b"]` needs an explicit
decision: explode into multiple rows, or encode as a set of indicator columns.
That decision belongs to you, not to a default.

---

## XML

```xml
<employees>
  <employee id="1"><name>John Doe</name><age>30</age></employee>
  <employee id="2"><name>Jane Smith</name><age>25</age></employee>
</employees>
```

```python
df = pd.read_xml("data.xml", xpath="//employee")   # columns: id, name, age
```

Still the native format of institutional and government data feeds. The `xpath`
argument is mandatory in practice — it selects the repeating element that
becomes a row. Attributes (`id`) and child elements (`name`) both become columns.

When the provider publishes an XSD or DTD schema, that is the contract: it
states which elements are required and what type each one holds.

---

## Text

```python
text = open("doc.txt", encoding="utf-8").read()
```

```python
with open("app.log", encoding="utf-8") as f:
    errors = [line for line in f if "ERROR" in line]
```

A text file has no structure beyond lines. `.read()` loads all of it; iterating
over the file object reads one line at a time, which is how a multi-gigabyte log
is filtered without holding it in memory.

---

## Images

```python
import numpy as np
from PIL import Image

img = Image.open("photo.jpg")
print(img.size, img.mode)          # (1920, 1080) RGB
arr = np.asarray(img)
print(arr.shape, arr.dtype)        # (1080, 1920, 3) uint8
```

`img.size` is (width, height); the array is (height, width, channels). Mixing
the two up is the classic first bug.

- **modes** — `RGB`, `RGBA`, `L` (grayscale), `CMYK`
- **compression** — JPEG is lossy, each re-save degrades it; PNG is lossless
- **EXIF metadata** — camera, date, often GPS position: personal data

---

## Unstructured data needs a manifest

For unstructured data the "loading" step is trivial and the *representation*
step is the entire problem.

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
