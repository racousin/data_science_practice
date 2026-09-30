"""MLP S6 — Aquarium Detection (file_v1).

Submission — `submission.csv`, `image_id,prediction_string`, one row per test
image: the format is `detection_metric.py`'s docstring. The file is handed out
byte for byte and imported here, so a student's local score is computed by the
same code as the leaderboard's.

Score — mAP@[.5:.95] over the 190 private test images (1,491 boxes), the
Ultralytics / COCO box metric. mAP@.5 and the AP of each class ride along.
A malformed file is rejected with a message naming the line, never scored.
"""
import os

from detection_metric import CLASS_NAMES, SubmissionError, evaluate, read_boxes

GROUND_TRUTH = "y_test.csv"


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), GROUND_TRUTH)
        self.ground_truth = read_boxes(path, with_confidence=False)

    def evaluate(self, submission_path):
        try:
            predictions = read_boxes(submission_path, with_confidence=True,
                                     image_ids=self.ground_truth.keys())
        except SubmissionError as exc:
            raise self.ParticipantSubmissionError(f"submission.csv rejected: {exc}") from None

        result = evaluate(self.ground_truth, predictions)
        detail = {k: round(v, 4) for k, v in result.items() if k != "n_detections"}
        detail["n_detections"] = result["n_detections"]
        worst = min(CLASS_NAMES, key=lambda c: result[f"ap_{c}"])
        message = (f"mAP50-95 {result['map50_95']:.4f}, mAP50 {result['map50']:.4f} over "
                   f"{len(self.ground_truth)} images, {result['n_detections']} detections; "
                   f"weakest class: {worst} (AP {result[f'ap_{worst}']:.3f}).")
        return {"agent_results": [{
            "agent_index": 0,
            "score": detail["map50_95"],
            "steps": 1,
            "metrics_detail": detail,
            "info_message": message,
        }]}
