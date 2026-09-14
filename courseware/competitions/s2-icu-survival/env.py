"""Session 2 — critical-care survival (file_v1). You do the preprocessing; the model is fixed.

Predict whether a seriously ill hospitalised patient died within 60 days of
study entry (target ``dead``), from what the hospital recorded on day 3. The
scorer never changes its model: it fits scikit-learn's default
``LogisticRegression()`` on the rows you submit, so every point of AUC comes
from your features.

Submission — ``submission.csv.gz``, a gzip-compressed CSV:

    id,feature_1,feature_2,...
    tr_3f9c2a71b0de,0.13,-1.20,...
    te_8d01c4e97a55,1.02,0.00,...

Written with ``features.to_csv("submission.csv.gz", index=False)``.

* Column ``id`` plus 1 to MAX_FEATURES feature columns, all numeric and finite.
* **Every test id** must appear.
* **Any subset of train ids** may appear, at least MIN_TRAIN_ROWS (4,000) of
  them: which rows to learn from is part of the preprocessing.
* Unknown ids, duplicated ids, duplicated or empty column names, text columns,
  NaN or inf, a file that is not gzip and a file that is not comma-separated
  are rejected, each with a message naming the first offender. A file with
  far more rows than there are ids is rejected before it is parsed. A rejected
  file scores 0 and never reaches the leaderboard. UTF-8 with or without a BOM.

Then, and nothing else — no scaling, no imputation, no tuning:

    model = LogisticRegression()                       # scikit-learn 1.8.0 defaults
    model.fit(X[train ids], y[train ids])              # the scorer's own labels
    score  = roc_auc_score(y_test,  model.predict_proba(X[test ids])[:, 1])
    score2 = roc_auc_score(y_train, model.predict_proba(X[train ids])[:, 1])

``info_message`` says so when lbfgs did not converge (unscaled features are the
usual cause) and when the train AUC is more than LEAK_GAP above the test AUC (a
feature that encodes the target on the train rows, typically a target encoding
fitted on the rows it encodes).

Only numpy, pandas, scikit-learn and the standard library.
"""
import csv
import gzip
import io
import os
import warnings
import zlib

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ID_COLUMN = "id"
TARGET_COLUMN = "dead"
SUBMISSION = "submission.csv.gz"
MAX_FEATURES = 300
MIN_TRAIN_ROWS = 4_000
LEAK_GAP = 0.05
# Read before parsing, so an oversized file is a message and not an OOM-killed
# pod. Every valid row is a distinct known id; the slack only lets a slightly
# fanned-out join reach the precise duplicate-id message.
ROW_SLACK = 1.1
MAX_DECOMPRESSED_BYTES = 2 * 1024 ** 3
HEADER_MAX_BYTES = 1024 ** 2
CHUNK_BYTES = 16 * 1024 ** 2
ENCODING = "utf-8-sig"          # utf-8, with or without the BOM Excel writes
METRIC_KEYS = ("auc", "auc_train", "n_features", "n_train_rows", "converged")


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        here = os.path.dirname(os.path.abspath(__file__))
        self.y_train = self._labels(os.path.join(here, "labels_train.csv"))
        self.y_test = self._labels(os.path.join(here, "labels_test.csv"))
        overlap = self.y_train.index.intersection(self.y_test.index)
        if len(overlap):
            raise ValueError(f"{len(overlap)} ids are in both label files, e.g. {overlap[0]!r}")

    @staticmethod
    def _labels(path):
        frame = pd.read_csv(path, dtype={ID_COLUMN: str})
        if list(frame.columns) != [ID_COLUMN, TARGET_COLUMN]:
            raise ValueError(f"{path}: expected columns {[ID_COLUMN, TARGET_COLUMN]}, "
                             f"found {list(frame.columns)}")
        labels = frame.set_index(ID_COLUMN)[TARGET_COLUMN]
        if not labels.index.is_unique or not labels.isin([0, 1]).all():
            raise ValueError(f"{path}: ids must be unique and labels 0/1")
        return labels.astype(np.int64)

    # ------------------------------------------------------------------ #
    def evaluate(self, submission_path):
        parsed, error = self._read(submission_path)
        if error is not None:
            return self._error(error)
        X_tr, y_tr, X_te, n_features = parsed
        y_te = self.y_test.to_numpy()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LogisticRegression().fit(X_tr, y_tr)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        auc_train = float(roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]))
        n_iter = int(np.max(model.n_iter_))

        notes = [f"ROC AUC {auc:.4f} on {len(y_te)} test rows; train AUC {auc_train:.4f} on "
                 f"{len(y_tr)} train rows; {n_features} features; lbfgs {n_iter} iterations"]
        if not converged:
            notes.append(f"WARNING: lbfgs did not converge in {n_iter} iterations — scale your "
                         f"features (columns on very different scales are the usual cause)")
        if auc_train - auc > LEAK_GAP:
            notes.append(f"WARNING: train AUC is {auc_train - auc:.3f} above test AUC — a feature "
                         f"encodes the target on the train rows (a target encoding fitted on the "
                         f"rows it encodes?)")
        return {"agent_results": [{
            "agent_index": 0,
            "score": round(auc, 6),
            "score2": round(auc_train, 6),
            "steps": int(len(y_te)),
            "info_message": ". ".join(notes),
            "metrics_detail": {
                "auc": round(auc, 6),
                "auc_train": round(auc_train, 6),
                "n_features": int(n_features),
                "n_train_rows": int(len(y_tr)),
                "converged": int(converged),
            },
        }]}

    # ------------------------------------------------------------------ #
    def _read(self, path):
        """Returns ((X_train, y_train, X_test, n_features), None) or (None, message)."""
        with open(path, "rb") as fh:
            magic = fh.read(2)
        if magic != b"\x1f\x8b":
            return None, (f"{SUBMISSION} is not gzip-compressed. Write it with "
                          f"features.to_csv('{SUBMISSION}', index=False).")

        n_ids = len(self.y_train) + len(self.y_test)
        max_rows = int(n_ids * ROW_SLACK)
        lines = size = 0
        last = b"\n"
        try:
            with gzip.open(path, "rb") as fh:
                while True:
                    chunk = fh.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    lines += chunk.count(b"\n")
                    size += len(chunk)
                    last = chunk[-1:]
                    if lines - 1 > max_rows:
                        return None, (f"{SUBMISSION} has more than {max_rows} rows, but there are only "
                                      f"{n_ids} ids ({len(self.y_train)} train + {len(self.y_test)} test): "
                                      f"ids are repeated or unknown — a join that fanned out?")
                    if size > MAX_DECOMPRESSED_BYTES:
                        return None, (f"{SUBMISSION} decompresses to more than "
                                      f"{MAX_DECOMPRESSED_BYTES // 1024 ** 3} GB; 100,000 rows of 300 "
                                      f"numbers take far less. Write plain numbers, not long text.")
            with gzip.open(path, "rt", encoding=ENCODING, newline="") as fh:
                first_line = fh.readline(HEADER_MAX_BYTES)
        except (OSError, EOFError, zlib.error, UnicodeDecodeError) as exc:
            return None, f"Could not read {SUBMISSION}: {exc}"
        if size and last != b"\n":
            lines += 1                                    # no newline after the last row
        if first_line and not first_line.endswith(("\n", "\r")) and lines > 1:
            return None, f"The header line of {SUBMISSION} is longer than 1 MB."
        header = next(csv.reader(io.StringIO(first_line)), None)
        if not header:
            return None, f"{SUBMISSION} is empty."
        if len(header) == 1 and header[0].count(";") > 0:
            return None, (f"{SUBMISSION} is separated by ';'. Use commas, to_csv's default "
                          f"(do not pass sep=';').")
        seen = set()
        for name in header:
            if not name.strip():
                return None, ("A column has an empty name — usually the DataFrame index, written "
                              "because to_csv was called without index=False.")
            if name in seen:
                return None, f"Column name {name!r} appears twice in the header; names must be unique."
            seen.add(name)
        if ID_COLUMN not in seen:
            return None, (f"Missing the '{ID_COLUMN}' column (header starts "
                          f"{[name[:40] for name in header[:5]]}).")
        n_features = len(header) - 1
        if n_features == 0:
            return None, f"No feature columns besides '{ID_COLUMN}'."
        if n_features > MAX_FEATURES:
            return None, (f"{n_features} feature columns; the limit is {MAX_FEATURES}. "
                          f"Select, don't pile up.")

        try:
            with gzip.open(path, "rt", encoding=ENCODING, newline="") as fh:
                df = pd.read_csv(fh, dtype={ID_COLUMN: str}, low_memory=False)
        except (OSError, EOFError, zlib.error, UnicodeDecodeError,
                pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            return None, f"Could not read {SUBMISSION} as a CSV: {exc}"
        if df.empty:
            return None, f"{SUBMISSION} has a header but no rows."

        ids = df[ID_COLUMN]
        if ids.isna().any():
            line = int(np.flatnonzero(ids.isna().to_numpy())[0]) + 2
            return None, f"Empty '{ID_COLUMN}' on line {line}."
        dup = ids.duplicated()
        if dup.any():
            first = int(np.flatnonzero(dup.to_numpy())[0])
            return None, (f"Duplicate id {ids.iloc[first]!r} (line {first + 2}): one row per id. "
                          f"{int(dup.sum())} duplicated row(s) — a join that fanned out?")
        is_train = ids.isin(self.y_train.index).to_numpy()
        is_test = ids.isin(self.y_test.index).to_numpy()
        unknown = ~(is_train | is_test)
        if unknown.any():
            first = int(np.flatnonzero(unknown)[0])
            return None, (f"Unknown id {ids.iloc[first]!r} (line {first + 2}): it is in neither "
                          f"train.csv nor test.csv. {int(unknown.sum())} unknown id(s).")
        # in test.csv order (labels_test.csv is written in that order), so the
        # id named is the first missing row of the file the student has
        missing = self.y_test.index[~self.y_test.index.isin(ids[is_test])]
        if len(missing):
            return None, (f"Test id {missing[0]!r} is missing: every id of test.csv must be "
                          f"present ({len(missing)} of {len(self.y_test)} missing).")
        n_train = int(is_train.sum())
        if n_train < MIN_TRAIN_ROWS:
            return None, (f"Only {n_train} train rows; submit at least {MIN_TRAIN_ROWS} of the "
                          f"{len(self.y_train)} ids of train.csv (you may drop the others).")

        features = [c for c in df.columns if c != ID_COLUMN]
        for col in features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                values = df[col]
                bad = pd.to_numeric(values, errors="coerce").isna() & values.notna()
                first = int(np.flatnonzero(bad.to_numpy())[0]) if bad.any() else 0
                return None, (f"Column {col!r} is not numeric: id {ids.iloc[first]!r} has "
                              f"{str(values.iloc[first])!r}. Encode text as numbers. "
                              f"({sum(not pd.api.types.is_numeric_dtype(df[c]) for c in features)} "
                              f"non-numeric column(s).)")

        values = df[features].to_numpy(dtype=np.float64)
        del df
        finite = np.isfinite(values)
        if not finite.all():
            rows, cols = np.nonzero(~finite)
            column = features[int(cols[0])]
            hint = (" That is the target column of train.csv, which the test rows do not have."
                    if column == TARGET_COLUMN else "")
            return None, (f"Column {column!r} is {values[rows[0], cols[0]]} for id "
                          f"{ids.iloc[int(rows[0])]!r}: LogisticRegression cannot take NaN or inf, "
                          f"impute them. {int((~finite).sum())} non-finite value(s).{hint}")
        del finite

        row_of = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
        train_ids = ids[is_train].to_numpy()
        X_tr = values[row_of.loc[train_ids].to_numpy()]
        y_tr = self.y_train.loc[train_ids].to_numpy()
        X_te = values[row_of.loc[self.y_test.index].to_numpy()]
        return (X_tr, y_tr, X_te, n_features), None

    @staticmethod
    def _error(message):
        return {"agent_results": [{
            "agent_index": 0,
            "score": 0.0,
            "steps": 0,
            "is_agent_code_error": True,
            "agent_code_error_message": message,
            "info_message": message,
            "metrics_detail": {k: 0 for k in METRIC_KEYS},
        }]}
