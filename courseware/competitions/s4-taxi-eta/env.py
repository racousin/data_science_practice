"""Session 4 — Taxi Arrival Promise (file_v1).

Promise an arrival time the ride beats 9 times out of 10. For each NYC
green-taxi trip in X_submission.csv, predict a duration in minutes; each
promise is scored with the pinball loss at tau = 0.9:

    loss = max(0.9 * (y - p), -0.1 * (y - p))

A minute late (the trip took longer than promised) costs 0.9, a minute of
padding 0.1. The promise that minimises the expected loss is the 90th
percentile of the trip's duration, so a calibrated submission keeps about 90%
of its promises.

Submission — `submission.csv`, one row per id of X_submission.csv:

    id,prediction
    te_75c7a09e8d17,18.4
    te_a10c9b861188,9.25

Ranking is on **-Pinball**: the mean pinball loss, negated. The leaderboard
sorts `mean_reward` descending and has no lower-is-better flag
(`modelmanager/modelmanager/competitions.py:214`), so the primary score has to
grow with quality — the s2-bike-demand -MAE precedent. 0 is perfect. Three
columns ride along for display:

    promise_kept_pct  trips that arrived at or before the promise, in %.
                      About 90 is calibrated; 100 means every promise was padded.
    avg_promise_min   the mean promised duration, in minutes.
    n_negative        predictions below zero. Scored, not rejected: a duration
                      cannot be negative, but a network with a linear output
                      layer can return one, and the count is how its author
                      finds out.

A malformed file is rejected, never imputed, with a message naming the line: a
header without an `id` or a `prediction` column, a header with no rows, an
empty, duplicate, missing or unknown id, a non-numeric value, NaN or infinity.
Other columns are ignored.

Pure standard library on purpose: the env image ships a full ML stack, but a
scorer that only needs `csv` and arithmetic has one less way to break.
"""
import csv
import math
import os

ID_COLUMN = "id"
TARGET_COLUMN = "prediction"
# The quantile the promise aims at: a late minute costs TAU, an early one 1 - TAU.
TAU = 0.9


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        path = os.path.join(os.path.dirname(__file__), "y_submission.csv")
        self.ground_truth = {}
        with open(path, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.ground_truth[row[ID_COLUMN]] = float(row[TARGET_COLUMN])

    def evaluate(self, submission_path):
        predictions, line_of, error = self._read_submission(submission_path)
        if error is not None:
            return self._error(error)

        missing = [k for k in self.ground_truth if k not in predictions]
        if missing:
            return self._error(
                f"Submission is missing {len(missing)} of {len(self.ground_truth)} "
                f"ids (e.g. {missing[0]!r}). Every id in X_submission.csv must "
                f"appear exactly once."
            )
        extra = [k for k in predictions if k not in self.ground_truth]
        if extra:
            return self._error(
                f"Line {line_of[extra[0]]}: id {extra[0]!r} is not in "
                f"X_submission.csv ({len(extra)} unknown id(s) in the file)."
            )

        n = len(self.ground_truth)
        loss = 0.0
        promised = 0.0
        kept = 0
        n_negative = 0
        for key, truth in self.ground_truth.items():
            promise = predictions[key]
            gap = truth - promise                     # > 0: the trip ran late
            loss += max(TAU * gap, (TAU - 1) * gap)
            promised += promise
            if truth <= promise:
                kept += 1
            if promise < 0:
                n_negative += 1
        pinball = loss / n
        kept_pct = 100.0 * kept / n
        avg_promise = promised / n

        note = (f"  ({n_negative} negative prediction(s): a duration cannot "
                f"be below zero)") if n_negative else ""
        return {"agent_results": [{
            "agent_index": 0,
            "score": round(-pinball, 6),
            "steps": n,
            "info_message": (
                f"-Pinball={-pinball:.4f}  promise kept {kept_pct:.1f}%  "
                f"avg promise {avg_promise:.2f} min  on {n} trips{note}"
            ),
            "metrics_detail": {
                "neg_pinball": round(-pinball, 6),
                "promise_kept_pct": round(kept_pct, 6),
                "avg_promise_min": round(avg_promise, 6),
                "n_negative": n_negative,
            },
        }]}

    def _read_submission(self, submission_path):
        """Returns (predictions, line_of_each_id, error). On success error is
        None; otherwise it is the message and the other two are None."""
        try:
            # utf-8-sig: a spreadsheet's byte-order mark is encoding, not a
            # malformed header.
            with open(submission_path, "r", newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    return None, None, "submission.csv is empty."
                names = [f.strip() for f in reader.fieldnames]
                for required in (ID_COLUMN, TARGET_COLUMN):
                    if names.count(required) != 1:
                        return None, None, (
                            f"Line 1: the header must name one '{ID_COLUMN}' "
                            f"column and one '{TARGET_COLUMN}' column (names are "
                            f"case-sensitive); found {reader.fieldnames}."
                        )
                # Read by the header's own spelling, so ' prediction' works.
                id_key = reader.fieldnames[names.index(ID_COLUMN)]
                target_key = reader.fieldnames[names.index(TARGET_COLUMN)]
                predictions, line_of = {}, {}
                for row in reader:
                    line_no = reader.line_num
                    key = (row.get(id_key) or "").strip()
                    raw = (row.get(target_key) or "").strip()
                    if not key:
                        return None, None, f"Line {line_no}: the '{ID_COLUMN}' is empty."
                    if key in predictions:
                        return None, None, (
                            f"Line {line_no}: duplicate id {key!r} (first on "
                            f"line {line_of[key]})."
                        )
                    try:
                        value = float(raw)
                    except ValueError:
                        return None, None, (
                            f"Line {line_no}: '{TARGET_COLUMN}' is {raw!r}, which "
                            f"is not a number. Predict the trip's duration in "
                            f"minutes as a real number."
                        )
                    if math.isnan(value) or math.isinf(value):
                        return None, None, (
                            f"Line {line_no}: '{TARGET_COLUMN}' is {raw!r}. NaN and "
                            f"infinity are not scoreable. A NaN usually means the "
                            f"training loss diverged — check the learning rate "
                            f"before checking the CSV."
                        )
                    predictions[key] = value
                    line_of[key] = line_no
        except UnicodeDecodeError:
            return None, None, "submission.csv is not valid UTF-8 text."
        except csv.Error as exc:
            return None, None, f"submission.csv is not a readable CSV file: {exc}"
        except OSError as exc:
            return None, None, f"Could not read submission.csv: {exc}"
        if not predictions:
            return None, None, "submission.csv has a header but no rows."
        return predictions, line_of, None

    @staticmethod
    def _error(message):
        return {"agent_results": [{
            "agent_index": 0,
            # 0.0 although -Pinball makes 0 the best possible score: a rejected
            # submission never reaches the leaderboard. This package ships
            # y_submission.csv, not y_test.csv, so the backend's upload check
            # only parses the CSV and THIS is the gate. It sets
            # is_agent_code_error, which fails the deployment (DEPLOY_FAILED),
            # and both leaderboard queries keep `status == ACTIVE` only
            # (`backend/app/views/leaderboard_helpers.py:131`,
            # `modelmanager/modelmanager/competitions.py:244`).
            "score": 0.0,
            "steps": 0,
            "is_agent_code_error": True,
            "agent_code_error_message": message,
            "info_message": message,
            "metrics_detail": {
                "neg_pinball": 0.0, "promise_kept_pct": 0.0,
                "avg_promise_min": 0.0, "n_negative": 0,
            },
        }]}
