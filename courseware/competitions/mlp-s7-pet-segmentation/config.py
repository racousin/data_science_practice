# Session 7 of MS2A - Machine Learning Practice ("Computer Vision 2"), course 15.
# Semantic segmentation on the Oxford-IIIT Pet dataset (CC BY-SA 4.0): which
# pixels of a 128 x 128 photo are the cat or dog. Lab 7's branch A already
# points students at OxfordIIITPet; this gives it a leaderboard.
#
# Why file_v1: the submission is a CSV of run-length-encoded masks; nothing
# the participant wrote runs. One row per image, so the platform's y_test.csv
# upload gate (header + id set) applies unchanged. The border band the
# annotators left undecided is not scored: it lives in a second private file,
# y_test_border.csv, because the gate requires y_test.csv to have exactly the
# submission's columns.
#
# Ranked on mean IoU over background and pet (DESC, the default order).
CONFIG = {
    "name": "MLP S7 — Pet Segmentation",
    "kernel_version": "file_v1",
    "module_slug": "s6-computer-vision-2",
    "label": "Pet Segmentation — a cat-or-dog mask for every photo: a U-Net from scratch, then a fine-tuned one, scored with mean IoU",
    "public_files": ["train.zip", "test_images.zip", "sample_submission.csv", "seg_metric.py"],
    # y_test.csv is both half the ground truth env.py reads and the file the
    # platform requires to start a file_v1 CSV challenge; env.py imports
    # seg_metric from next to itself — the same bytes students get.
    "private_files": ["y_test.csv", "y_test_border.csv", "seg_metric.py"],
    "benchmark_file": "data/benchmark_submission.csv",
    # reference_solution.py: the starter notebook's SmallUNet(c=16), 15 epochs,
    # Adam one-cycle 3e-3, batch 32, seed 0, on 4,972 of the 5,849 training
    # images (Apple M4 GPU), 2026-10-05. The scorer is deterministic, so the
    # file scores this exactly; the training run is not reproducible across
    # machines. Val mIoU 0.9226 at epoch 15. The notebook's fine-tuned
    # smp.Unet("resnet18", imagenet), 8 epochs at 1e-3, scores 0.9565 on the
    # same test set; the empty sample submission 0.3307.
    "benchmark_expected_score": 0.9149,
    "benchmark_score_tol": 1e-4,
    # The module's bar, below the benchmark: a rerun of the notebook on a T4
    # lands near 0.915, not on it. One run measured: the headroom is a judgement.
    "pass_threshold": 0.90,
    "dataset_label": "Oxford-IIIT Pet — segmentation",
    "dataset_description": (
        "Oxford-IIIT Pet (Parkhi et al. 2012, CC BY-SA 4.0), resized to 128 x 128. "
        "train.zip: 5,849 photos (images/<id>.jpg) and masks (masks/<id>.png: "
        "0 background, 1 pet, 2 border, not scored). test_images.zip: 1,500 photos "
        "to segment. seg_metric.py is the leaderboard's RLE format and mean IoU; "
        "sample_submission.csv shows the format."
    ),
    "simulation_timeout_sec": 120,
    # Deterministic scorer: the same CSV always yields the same mIoU.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    # Every field is required by the settings route (backend _metric_spec.MetricSpec).
    "metrics": [
        {"key": "miou", "label": "mIoU", "source": "env", "agg": "mean", "order": "desc",
         "format": "number", "unit": None, "precision": 4, "is_ranking": True, "visible": True},
        {"key": "iou_pet", "label": "IoU pet", "source": "env", "agg": "mean", "order": "desc",
         "format": "number", "unit": None, "precision": 4, "is_ranking": False, "visible": True},
        {"key": "iou_background", "label": "IoU background", "source": "env", "agg": "mean", "order": "desc",
         "format": "number", "unit": None, "precision": 4, "is_ranking": False, "visible": True},
        {"key": "dice_pet", "label": "Dice pet", "source": "env", "agg": "mean", "order": "desc",
         "format": "number", "unit": None, "precision": 4, "is_ranking": False, "visible": True},
        {"key": "pixel_accuracy", "label": "Pixel accuracy", "source": "env", "agg": "mean", "order": "desc",
         "format": "number", "unit": None, "precision": 4, "is_ranking": False, "visible": True},
    ],
    "is_public_initial": False,
}
