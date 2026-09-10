"""Session 4 — MNIST warm-up (file_v1).

The dry run for the project submission path. Lab 4 trains a packaged, tested
MLP; this competition takes its predictions on 5,000 held-out digits.

Submission — `submission.csv`, one row per test id:

    id,label
    te_00000,7
    te_00001,2

`label` is the predicted digit, 0-9. Every test id must appear exactly once.

Score is plain accuracy; the leaderboard also carries macro-F1, which is the
one that notices when a model has quietly stopped predicting a whole class.

A note on what this measures. The images come from public MNIST, so the labels
exist somewhere on the internet — this is a *plumbing* check, deliberately, and
Lab 4 says as much: getting on the board matters, your position does not. What
it proves is that your training pipeline produces a file the platform accepts,
before that matters for the project grade.

Pure standard library: the env image ships a full ML stack, but a scorer that
needs only `csv` and arithmetic has one less way to break.
"""
import csv
import os

ID_COLUMN = "id"
TARGET_COLUMN = "label"
CLASSES = [str(d) for d in range(10)]
VALID = set(CLASSES)


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        path = os.path.join(os.path.dirname(__file__), "y_submission.csv")
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
                f"test ids (e.g. {missing[0]!r}). Every id in X_submission.csv must "
                f"appear exactly once."
            )
        extra = [k for k in predictions if k not in self.ground_truth]
        if extra:
            return self._error(
                f"Submission has {len(extra)} id(s) that are not in the test set "
                f"(e.g. {extra[0]!r})."
            )

        total = len(self.ground_truth)
        correct = 0
        tp = {c: 0 for c in CLASSES}
        predicted_count = {c: 0 for c in CLASSES}
        actual_count = {c: 0 for c in CLASSES}
        for key, truth in self.ground_truth.items():
            predicted = predictions[key]
            actual_count[truth] += 1
            predicted_count[predicted] += 1
            if predicted == truth:
                correct += 1
                tp[truth] += 1

        accuracy = correct / total if total else 0.0
        f1s = []
        for c in CLASSES:
            precision = tp[c] / predicted_count[c] if predicted_count[c] else 0.0
            recall = tp[c] / actual_count[c] if actual_count[c] else 0.0
            f1s.append(2 * precision * recall / (precision + recall)
                       if (precision + recall) else 0.0)
        macro_f1 = sum(f1s) / len(f1s)
        worst_class = min(range(len(CLASSES)), key=lambda i: f1s[i])

        return {"agent_results": [{
            "agent_index": 0,
            "score": round(accuracy, 6),
            "score2": round(macro_f1, 6),
            "steps": total,
            "info_message": (
                f"accuracy={accuracy:.4f} ({correct}/{total})  "
                f"macro-F1={macro_f1:.4f}  "
                f"weakest digit={CLASSES[worst_class]} (F1={f1s[worst_class]:.4f})"
            ),
            "metrics_detail": {
                "accuracy": round(accuracy, 6),
                "macro_f1": round(macro_f1, 6),
                "worst_class_f1": round(f1s[worst_class], 6),
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
                            f"expected a digit 0-9. Write the predicted class, "
                            f"not a probability or a one-hot row."
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
                "accuracy": 0.0, "macro_f1": 0.0, "worst_class_f1": 0.0,
            },
        }]}
