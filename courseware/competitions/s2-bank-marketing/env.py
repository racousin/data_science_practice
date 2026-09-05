"""Session 2 — Bank term deposit (file_v1).

Predict whether a client subscribes to a term deposit after a direct marketing
call. The classification half of Session 2: mixed numeric and categorical
columns, and **11.7% positive** — which is the whole reason the ranking metric
is F1 and not accuracy.

Submission — `submission.csv`, one row per test id:

    id,prediction
    te_00000,0
    te_00001,1

`prediction` is 1 for "subscribed" and 0 otherwise. Every test id must appear
exactly once; extra ids are an error, and so is any value other than 0 or 1 —
a probability is rejected rather than silently thresholded, because a scorer
that guessed your threshold for you would be reporting its own choice, not
yours.

Primary score is F1 on the positive class. Always predicting 0 scores 88.3%
accuracy and an F1 of 0.00, which is the entire argument. Accuracy, precision
and recall are on the leaderboard too, so the tradeoff your threshold made is
visible.

Pure standard library on purpose: the env image ships a full ML stack, but a
scorer that only needs `csv` and arithmetic has one less way to break.
"""
import csv
import os

ID_COLUMN = "id"
TARGET_COLUMN = "prediction"
VALID = {"0", "1"}


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        path = os.path.join(os.path.dirname(__file__), "y_test.csv")
        self.ground_truth = {}
        with open(path, "r", newline="") as fh:
            for row in csv.DictReader(fh):
                self.ground_truth[row[ID_COLUMN]] = row[TARGET_COLUMN].strip()

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

        tp = fp = fn = tn = 0
        for key, truth in self.ground_truth.items():
            predicted = predictions[key]
            if predicted == "1" and truth == "1":
                tp += 1
            elif predicted == "1":
                fp += 1
            elif truth == "1":
                fn += 1
            else:
                tn += 1

        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        accuracy = (tp + tn) / total if total else 0.0

        return {"agent_results": [{
            "agent_index": 0,
            "score": round(f1, 6),
            "score2": round(accuracy, 6),
            "steps": total,
            "info_message": (
                f"F1={f1:.4f}  accuracy={accuracy:.4f}  "
                f"precision={precision:.4f}  recall={recall:.4f}  "
                f"(tp={tp} fp={fp} fn={fn} tn={tn})"
            ),
            "metrics_detail": {
                "f1": round(f1, 6),
                "accuracy": round(accuracy, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
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
                    value = (row.get(TARGET_COLUMN) or "").strip()
                    if not key:
                        return None, f"Empty '{ID_COLUMN}' on line {line_no}."
                    if key in predictions:
                        return None, f"Duplicate id {key!r} on line {line_no}."
                    if value not in VALID:
                        return None, (
                            f"Line {line_no}: '{TARGET_COLUMN}' is {value!r}; "
                            f"expected 0 or 1 (1 means subscribed)."
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
                "f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            },
        }]}
