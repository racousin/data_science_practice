"""Session 2 — Bike sharing demand (file_v1).

Predict the hourly rental count from calendar and weather columns. The first
challenge of the course: read a table, look at it, patch the holes in it, fit a
linear model, submit.

The split is **chronological** — the last 20% of the hours are held out, so this
is a forecast and not an interpolation. The features carry missing values on
purpose; the scorer never sees them, but nothing fits until they are dealt with.

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
that, and it keeps MAE's units — -138.88 reads as "wrong by 138.88 bikes an
hour on average", and 0 is perfect. RMSE rides along for display.

R2 was reported alongside until 2026-09-08 and is gone. Two numbers measuring
the same residuals invite the student to quote whichever is kinder, which is the
one habit `evaluation-metrics` (now Session 2) exists to break; and R2 is
measured against the variance of the *test* window, so under a chronological
split it compares a forecast to a mean the forecaster could not have known.

Pure standard library on purpose: the env image ships a full ML stack, but a
scorer that only needs `csv` and arithmetic has one less way to break.
"""
import csv
import math
import os

ID_COLUMN = "id"
TARGET_COLUMN = "prediction"


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
        errors = [predictions[k] - self.ground_truth[k] for k in self.ground_truth]

        rmse = math.sqrt(sum(e * e for e in errors) / n)
        mae = sum(abs(e) for e in errors) / n
        n_negative = sum(1 for k in self.ground_truth if predictions[k] < 0)

        note = f"  ({n_negative} negative prediction(s))" if n_negative else ""
        return {"agent_results": [{
            "agent_index": 0,
            "score": round(-mae, 6),
            "score2": round(rmse, 6),
            "steps": n,
            "info_message": (
                f"-MAE={-mae:.2f}  RMSE={rmse:.2f}  on {n} test rows{note}"
            ),
            "metrics_detail": {
                "neg_mae": round(-mae, 6),
                "rmse": round(rmse, 6),
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
            # 0.0 despite -MAE making zero the *best* possible score: a
            # rejected submission never reaches the leaderboard to be ranked.
            # Upload validation stops a malformed CSV before deployment
            # (`backend/app/services/check_upload_files/check_csv_submission.py`),
            # and anything that gets past it and errors here fails the
            # deployment (`serviceapiclient/job/deployment.py:298`
            # check_deployment_completion -> DEPLOY_FAILED), which both
            # leaderboard queries exclude by `status == ACTIVE`
            # (`backend/app/views/leaderboard_helpers.py:131`,
            # `modelmanager/modelmanager/competitions.py:244`). Verified live.
            "score": 0.0,
            "steps": 0,
            "is_agent_code_error": True,
            "agent_code_error_message": message,
            "info_message": message,
            "metrics_detail": {
                "neg_mae": 0.0, "rmse": 0.0, "n_negative": 0,
            },
        }]}
