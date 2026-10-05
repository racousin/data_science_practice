"""MLP S7 — Pet Segmentation (file_v1).

Submission — `submission.csv`, `image_id,mask_rle`, one row per test image:
the format is `seg_metric.py`'s docstring. The file is handed out byte for
byte and imported here, so a student's local score is computed by the same
code as the leaderboard's.

Score — mean IoU over background and pet, over every scored pixel of the 1,500
private test images. The border band around each pet (`y_test_border.csv`) is
not scored. A malformed file is rejected with a message naming the line.
"""
import os

from seg_metric import BORDER, SubmissionError, evaluate, read_rle_csv

HERE = os.path.dirname(os.path.abspath(__file__))


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        pet = read_rle_csv(os.path.join(HERE, "y_test.csv"))
        border = read_rle_csv(os.path.join(HERE, "y_test_border.csv"), image_ids=pet.keys(),
                              column="border_rle")
        self.ground_truth = {}
        for iid, mask in pet.items():
            labels = mask.astype("uint8")
            labels[border[iid]] = BORDER
            self.ground_truth[iid] = labels

    def evaluate(self, submission_path):
        try:
            predictions = read_rle_csv(submission_path, image_ids=self.ground_truth.keys())
        except SubmissionError as exc:
            raise self.ParticipantSubmissionError(f"submission.csv rejected: {exc}") from None

        result = {k: round(float(v), 4) for k, v in evaluate(self.ground_truth, predictions).items()}
        message = (f"mIoU {result['miou']:.4f} (pet {result['iou_pet']:.4f}, background "
                   f"{result['iou_background']:.4f}), Dice {result['dice_pet']:.4f} "
                   f"over {len(self.ground_truth)} images.")
        return {"agent_results": [{
            "agent_index": 0,
            "score": result["miou"],
            "steps": 1,
            "metrics_detail": result,
            "info_message": message,
        }]}
