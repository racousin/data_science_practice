"""Session 4 — Forest cover type (file_v1).

Predict which of seven tree species dominates a 30x30 m patch of the Roosevelt
National Forest, from 54 cartographic features. 14,000 training rows, 3,500 to
predict, seven classes balanced by construction.

This is the challenge the student writes unaided, after the California-housing
notebook has shown the whole PyTorch loop worked out. Same argument, different
target type: logistic regression scores 0.696 accuracy here and a 54-128-64-7
MLP scores 0.814.

Submission — `submission.csv`, one row per test id:

    id,prediction
    te_00000,2
    te_00001,7

`prediction` is the cover type, an integer 1-7:

    1 Spruce/Fir        3 Ponderosa Pine     5 Aspen        7 Krummholz
    2 Lodgepole Pine    4 Cottonwood/Willow  6 Douglas-fir

Note the 1-based coding. `torch.nn.CrossEntropyLoss` wants 0-based indices, so
the trip through the network is `label - 1` going in and `argmax + 1` coming
out. Submitting a 0 is rejected with a message saying exactly that, because a
whole column shifted by one is otherwise a silent 0.14-accuracy mystery.

Score is plain accuracy, which is meaningful here only because the classes are
balanced 1/7 by construction — chance is 0.1429 and accuracy equals balanced
accuracy. Macro-F1 rides along, and so does the worst class's F1, which is the
column that notices a model has quietly stopped predicting a cover type
altogether.

Pure standard library on purpose: the env image ships a full ML stack, but a
scorer that only needs `csv` and arithmetic has one less way to break.
"""
import csv
import os

ID_COLUMN = "id"
TARGET_COLUMN = "prediction"
CLASSES = [str(c) for c in range(1, 8)]
VALID = set(CLASSES)
NAMES = {
    "1": "Spruce/Fir", "2": "Lodgepole Pine", "3": "Ponderosa Pine",
    "4": "Cottonwood/Willow", "5": "Aspen", "6": "Douglas-fir",
    "7": "Krummholz",
}


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
        worst = min(range(len(CLASSES)), key=lambda i: f1s[i])
        never_predicted = [NAMES[c] for c in CLASSES if not predicted_count[c]]

        note = ""
        if never_predicted:
            note = ("  NEVER PREDICTED: " + ", ".join(never_predicted)
                    + " — the head has collapsed onto the easy classes")
        return {"agent_results": [{
            "agent_index": 0,
            "score": round(accuracy, 6),
            "score2": round(macro_f1, 6),
            "steps": total,
            "info_message": (
                f"accuracy={accuracy:.4f} ({correct}/{total})  "
                f"macro-F1={macro_f1:.4f}  "
                f"weakest={NAMES[CLASSES[worst]]} (F1={f1s[worst]:.4f}){note}"
            ),
            "metrics_detail": {
                "accuracy": round(accuracy, 6),
                "macro_f1": round(macro_f1, 6),
                "worst_class_f1": round(f1s[worst], 6),
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
                    if value == "0":
                        return None, (
                            f"Line {line_no}: '{TARGET_COLUMN}' is '0'. Cover "
                            f"types are 1-7, not 0-6. CrossEntropyLoss works in "
                            f"0-based indices, so add 1 back after argmax: "
                            f"`logits.argmax(dim=1) + 1`."
                        )
                    if value not in VALID:
                        return None, (
                            f"Line {line_no}: '{TARGET_COLUMN}' is {value!r}; "
                            f"expected a cover type 1-7. Write the predicted "
                            f"class, not a probability and not a row of logits."
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
