"""Session 4 — California housing (file_v1).

Predict the median house value of a California census district from eight
numeric features. 16,512 training rows, 4,128 to predict.

This is the challenge Session 4's worked notebook builds, and it exists to show
that the machinery of Sessions 2 and 3 has a ceiling. Plain linear regression
scores R2 = 0.576 here; an 8-64-64-1 MLP trained in PyTorch scores 0.776. The
relationship is genuinely non-linear — latitude and longitude alone are two
numbers that mean nothing apart and a great deal together — so the extra
capacity buys generalisation rather than memorisation.

Submission — `submission.csv`, one row per test id:

    id,prediction
    te_00000,1.943
    te_00001,3.108

`prediction` is the median house value in units of $100,000. The target is
capped at 5.00001 and floored at 0.14999 in the source data, so anything
outside that interval is a prediction the dataset cannot contain — the scorer
counts those rather than clipping them, because noticing is the lesson.

Ranking is on **R2**. The leaderboard sorts descending
(`modelmanager/modelmanager/competitions.py:214`), so the primary score has to
be higher-is-better; R2 = 0 is exactly the constant model that predicts the
training mean. RMSE and MAE are shown alongside, in units of $100,000.

Pure standard library on purpose: the env image ships a full ML stack, but a
scorer that only needs `csv` and arithmetic has one less way to break.
"""
import csv
import math
import os

ID_COLUMN = "id"
TARGET_COLUMN = "prediction"

# The documented range of the source target, not a property of the held-out
# split: `fetch_california_housing` floors at 0.14999 and caps at 5.00001 over
# all 20,640 districts. Reporting how far a submission leaves it tells a
# competitor something actionable without revealing a label.
TARGET_MIN = 0.14999
TARGET_MAX = 5.00001


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        path = os.path.join(os.path.dirname(__file__), "y_submission.csv")
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
                f"test ids (e.g. {missing[0]!r}). Every id in X_submission.csv must "
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
        n_out_of_range = sum(
            1 for k in self.ground_truth
            if not (TARGET_MIN <= predictions[k] <= TARGET_MAX)
        )

        note = ""
        if n_out_of_range:
            note = (f"  ({n_out_of_range} prediction(s) outside the target's "
                    f"[{TARGET_MIN}, {TARGET_MAX}] range — clipping is free score)")
        return {"agent_results": [{
            "agent_index": 0,
            "score": round(r2, 6),
            "score2": round(rmse, 6),
            "steps": n,
            "info_message": (
                f"R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
                f"on {n} test districts{note}"
            ),
            "metrics_detail": {
                "r2": round(r2, 6),
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "n_out_of_range": n_out_of_range,
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
                            f"is not a number. Predict the median house value as "
                            f"a real number in units of $100,000."
                        )
                    if math.isnan(value) or math.isinf(value):
                        return None, (
                            f"Line {line_no}: '{TARGET_COLUMN}' is {raw!r}. NaN and "
                            f"infinity are not scoreable. A NaN here usually means "
                            f"the training loss diverged — check the learning rate "
                            f"before checking the CSV."
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
            "score": 0.0,
            "steps": 0,
            "is_agent_code_error": True,
            "agent_code_error_message": message,
            "info_message": message,
            "metrics_detail": {
                "r2": 0.0, "rmse": 0.0, "mae": 0.0, "n_out_of_range": 0,
            },
        }]}
