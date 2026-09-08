"""Session 2 — Bike sharing demand (file_v1).

Predict the hourly rental count from calendar and weather columns. The first
challenge of the course: read a table, look at it, fit a linear model, submit.

Submission — `submission.csv`, one row per test id:

    id,prediction
    te_75c7a09e8d17,143.2
    te_a10c9b861188,88.0

`prediction` is a real number. Counts are non-negative, but a plain linear
model does return negatives on the low hours and that is not rejected — it is
scored, and the score is what tells you it happened.

Ranking is on **-MAE** — the mean absolute error, negated. The leaderboard
sorts `mean_reward` descending and has no lower-is-better flag
(`modelmanager/modelmanager/competitions.py:210-213`), so the primary score has
to increase with quality; negating an error metric is the direct way to get
that, and it keeps MAE's units — -103.74 reads as "wrong by 103.74 bikes an
hour on average", and 0 is perfect. RMSE and R2 ride along for display.

Pure standard library on purpose: the env image ships a full ML stack, but a
scorer that only needs `csv` and arithmetic has one less way to break.
"""
import csv
import math
import os

ID_COLUMN = "id"
TARGET_COLUMN = "prediction"

# A rejected submission must not out-rank a real model. Under R2 an error could
# score 0.0 and land mid-table, at the mean model. Under -MAE 0.0 is a *perfect*
# score, and the leaderboard orders on mean_reward DESC with no filter on
# is_agent_code_error (`modelmanager/modelmanager/competitions.py:210-213`) — so
# a rejection scoring 0.0 would sit at the top of the board. Predicting a
# constant zero for every hour scores about -189; nothing honest comes near this.
ERROR_SCORE = -1e9


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        path = os.path.join(os.path.dirname(__file__), "y_test.csv")
        self.ground_truth = {}
        with open(path, "r", newline="") as fh:
            for row in csv.DictReader(fh):
                self.ground_truth[row[ID_COLUMN]] = float(row[TARGET_COLUMN])

    def evaluate(self, submission_path):
        predictions, error = self._read_submission(submission_path)
        if error is not None:
            return self._error(error)

        missing = [k for k in self.ground_truth if k not in predictions]
        if missing:
            return self._error(
                f"Submission is missing {len(missing)} of {len(self.ground_truth)} "
                f"test ids (e.g. {missing[0]!r}). Every id in X_test.csv must "
                f"appear exactly once."
            )
        extra = [k for k in predictions if k not in self.ground_truth]
        if extra:
            return self._error(
                f"Submission has {len(extra)} id(s) that are not in the test set "
                f"(e.g. {extra[0]!r})."
            )

        n = len(self.ground_truth)
        truths = [self.ground_truth[k] for k in self.ground_truth]
        errors = [predictions[k] - self.ground_truth[k] for k in self.ground_truth]
        mean_truth = sum(truths) / n

        ss_res = sum(e * e for e in errors)
        ss_tot = sum((t - mean_truth) ** 2 for t in truths)
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        rmse = math.sqrt(ss_res / n)
        mae = sum(abs(e) for e in errors) / n
        n_negative = sum(1 for k in self.ground_truth if predictions[k] < 0)

        note = f"  ({n_negative} negative prediction(s))" if n_negative else ""
        return {"agent_results": [{
            "agent_index": 0,
            "score": round(-mae, 6),
            "score2": round(rmse, 6),
            "steps": n,
            "info_message": (
                f"-MAE={-mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}  "
                f"on {n} test rows{note}"
            ),
            "metrics_detail": {
                "neg_mae": round(-mae, 6),
                "rmse": round(rmse, 6),
                "r2": round(r2, 6),
                "n_negative": n_negative,
            },
        }]}

    def _read_submission(self, submission_path):
        """Returns (predictions, error_message); exactly one is meaningful."""
        try:
            with open(submission_path, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    return None, "submission.csv is empty."
                fields = [f.strip() for f in reader.fieldnames]
                if ID_COLUMN not in fields or TARGET_COLUMN not in fields:
                    return None, (
                        f"submission.csv must have columns '{ID_COLUMN}' and "
                        f"'{TARGET_COLUMN}'; found {reader.fieldnames}."
                    )
                predictions = {}
                for line_no, row in enumerate(reader, start=2):
                    key = (row.get(ID_COLUMN) or "").strip()
                    raw = (row.get(TARGET_COLUMN) or "").strip()
                    if not key:
                        return None, f"Empty '{ID_COLUMN}' on line {line_no}."
                    if key in predictions:
                        return None, f"Duplicate id {key!r} on line {line_no}."
                    try:
                        value = float(raw)
                    except ValueError:
                        return None, (
                            f"Line {line_no}: '{TARGET_COLUMN}' is {raw!r}, which "
                            f"is not a number. Predict the hourly count as a "
                            f"real number."
                        )
                    if math.isnan(value) or math.isinf(value):
                        return None, (
                            f"Line {line_no}: '{TARGET_COLUMN}' is {raw!r}. NaN and "
                            f"infinity are not scoreable — check for missing "
                            f"values in the rows you predicted."
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
            "score": ERROR_SCORE,
            "steps": 0,
            "is_agent_code_error": True,
            "agent_code_error_message": message,
            "info_message": message,
            "metrics_detail": {
                "neg_mae": ERROR_SCORE, "rmse": 0.0, "r2": 0.0, "n_negative": 0,
            },
        }]}
