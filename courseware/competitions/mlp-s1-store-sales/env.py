"""MLP S1 — Multi-Source Store Sales (file_v1).

Predict `quantity_sold` for every Neighborhood_Market item, from the sales of
four other stores spread over files, an API, a scraped page and the course
database. The data is the Data Science Practice module 4 exercise, unchanged;
`prepare_data.py` explains where the hidden target comes from.

Submission — `submission.csv`, one row per item of Neighborhood_Market_data.csv
(409 rows), in any order:

    item_code,quantity_sold
    P0002,164.3
    P0004,251.0

`quantity_sold` is a real number. Negative values are scored, not rejected.

Ranking is on **-MAE**: the mean absolute error, negated. The leaderboard
sorts `mean_reward` descending and has no lower-is-better flag
(`modelmanager/modelmanager/competitions.py:210-213`), so the primary score has
to grow with quality — the s2-bike-demand precedent. -12.5 reads "wrong by 12.5
units per item on average", and 0 is perfect. RMSE and the number of scored
rows ride along for display.

A malformed file is rejected, never imputed, with a message that names the
problem: no `item_code` or `quantity_sold` column, a header with no rows, an
empty or duplicate item_code, a non-numeric value, NaN or infinity, a missing
item, an unknown item. Other columns are ignored.

The ground truth is read from `neighborhood_market_target.csv`. The same bytes
are also uploaded as `y_test.csv`, the name the platform requires before it
starts a CSV challenge and reads its upload check from; this file does not read
that copy.

Pure standard library on purpose: the env image ships a full ML stack, but a
scorer that only needs `csv` and arithmetic has one less way to break.
"""
import csv
import math
import os

ID_COLUMN = "item_code"
TARGET_COLUMN = "quantity_sold"
TARGET_FILE = "neighborhood_market_target.csv"


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        path = os.path.join(os.path.dirname(__file__), TARGET_FILE)
        self.ground_truth = {}
        with open(path, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.ground_truth[row[ID_COLUMN]] = float(row[TARGET_COLUMN])
        if not self.ground_truth:
            raise RuntimeError(f"{TARGET_FILE} has no rows")

    def evaluate(self, submission_path):
        predictions, error = self._read_submission(submission_path)
        if error is not None:
            return self._error(error)

        missing = [k for k in self.ground_truth if k not in predictions]
        if missing:
            return self._error(
                f"submission.csv is missing {len(missing)} of "
                f"{len(self.ground_truth)} Neighborhood_Market items (e.g. "
                f"{missing[0]!r}). Every item_code in "
                f"Neighborhood_Market_data.csv must appear exactly once."
            )
        unknown = [k for k in predictions if k not in self.ground_truth]
        if unknown:
            return self._error(
                f"submission.csv has {len(unknown)} item_code(s) that are not "
                f"Neighborhood_Market items (e.g. {unknown[0]!r}). Predict only "
                f"the items of Neighborhood_Market_data.csv, not the four "
                f"training stores."
            )

        n = len(self.ground_truth)
        errors = [predictions[k] - self.ground_truth[k]
                  for k in self.ground_truth]
        mae = sum(abs(e) for e in errors) / n
        rmse = math.sqrt(sum(e * e for e in errors) / n)

        return {"agent_results": [{
            "agent_index": 0,
            "score": round(-mae, 6),
            "score2": round(rmse, 6),
            "steps": n,
            "info_message": (
                f"-MAE={-mae:.2f}  RMSE={rmse:.2f}  on {n} items"
            ),
            "metrics_detail": {
                "neg_mae": round(-mae, 6),
                "rmse": round(rmse, 6),
                "n_rows": n,
            },
        }]}

    def _read_submission(self, submission_path):
        """Returns (predictions, error_message); exactly one is meaningful."""
        try:
            with open(submission_path, "r", newline="",
                      encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    return None, "submission.csv is empty."
                fields = [f.strip() for f in reader.fieldnames]
                absent = [c for c in (ID_COLUMN, TARGET_COLUMN)
                          if c not in fields]
                if absent:
                    return None, (
                        f"submission.csv must have the columns '{ID_COLUMN}' "
                        f"and '{TARGET_COLUMN}'; missing {absent}, found "
                        f"{fields}. Write it with "
                        f"df[['item_code', 'quantity_sold']].to_csv("
                        f"'submission.csv', index=False)."
                    )
                reader.fieldnames = fields
                predictions = {}
                for line_no, row in enumerate(reader, start=2):
                    key = (row.get(ID_COLUMN) or "").strip()
                    raw = (row.get(TARGET_COLUMN) or "").strip()
                    if not key:
                        return None, f"Empty '{ID_COLUMN}' on line {line_no}."
                    if key in predictions:
                        return None, (
                            f"Duplicate item_code {key!r} on line {line_no}. "
                            f"Each item appears exactly once."
                        )
                    try:
                        value = float(raw)
                    except ValueError:
                        return None, (
                            f"Line {line_no} ({key}): '{TARGET_COLUMN}' is "
                            f"{raw!r}, which is not a number."
                        )
                    if math.isnan(value) or math.isinf(value):
                        return None, (
                            f"Line {line_no} ({key}): '{TARGET_COLUMN}' is "
                            f"{raw!r}. NaN and infinity cannot be scored — "
                            f"check for missing values in the features you "
                            f"predicted from."
                        )
                    predictions[key] = value
        except UnicodeDecodeError:
            return None, "submission.csv is not valid UTF-8 text."
        except OSError as exc:
            return None, f"Could not read submission.csv: {exc}"
        if not predictions:
            return None, "submission.csv has a header but no rows."
        return predictions, None

    @staticmethod
    def _error(message):
        return {"agent_results": [{
            "agent_index": 0,
            # 0.0 although -MAE makes zero the best possible score: a rejected
            # submission never reaches the leaderboard to be ranked. See
            # competitions/README.md "Ranking direction" — upload validation
            # stops a wrong header or id set, and an env error sets
            # is_agent_code_error, which fails the deployment; both
            # leaderboard queries list ACTIVE agents only.
            "score": 0.0,
            "steps": 0,
            "is_agent_code_error": True,
            "agent_code_error_message": message,
            "info_message": message,
            "metrics_detail": {"neg_mae": 0.0, "rmse": 0.0, "n_rows": 0},
        }]}
