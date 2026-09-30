# Session 6 of MS2A - Machine Learning Practice ("Computer Vision 2"), course 15.
# Object detection on the Roboflow Aquarium dataset (CC BY 4.0), after Day 3
# TP2 of ai_for_sciences: fine-tune a COCO-pretrained YOLO on 448 labelled
# images and detect 7 classes of marine animals in 190 private test images.
# It replaces Gymnasium · CarRacing-v3 (47), a control task whose
# prerequisites are Sessions 9-10 (COURSE_STATE.md §4).
#
# Why file_v1: the submission is a CSV of boxes; nothing the participant wrote
# runs. The format is one row per image with a prediction string, so the
# platform's y_test.csv upload gate (header + id set) applies unchanged.
#
# Ranked on mAP@[.5:.95] (DESC, the default order), the metric Ultralytics
# prints as mAP50-95 — detection_metric.py reproduces it, and
# reference_solution.py checks the two agree on held-out images.
CONFIG = {
    "name": "MLP S6 — Aquarium Detection",
    "kernel_version": "file_v1",
    "module_slug": "s6-computer-vision-2",
    "label": "Aquarium Detection — seven marine species, boxes and classes: fine-tune a pretrained YOLO, scored with mAP50-95",
    "metric": "map50_95",
    "public_files": ["train.zip", "test_images.zip", "classes.csv",
                     "sample_submission.csv", "detection_metric.py"],
    # y_test.csv is both the ground truth env.py reads and the file the platform
    # requires to start a file_v1 CSV challenge; its header and ids become the
    # upload gate. env.py imports detection_metric from next to itself — the
    # same bytes students get.
    "private_files": ["y_test.csv", "detection_metric.py"],
    "benchmark_file": "data/benchmark_submission.csv",
    # reference_solution.py: YOLOv8n, 30 epochs, imgsz 640, batch 16, seed 0,
    # on 381 of the 448 images (trained on an Apple M4 GPU, predicted on CPU,
    # conf 0.001), 2026-09-30. mAP50 0.6037; weakest class puffin, AP 0.088.
    # The scorer is deterministic, so the file scores this exactly; the
    # training run that made it is not reproducible across machines.
    "benchmark_expected_score": 0.3346,
    "benchmark_score_tol": 1e-4,
    # The module's bar, below the benchmark: the notebook's recipe is not
    # reproducible across machines, so a student's rerun lands around the
    # benchmark rather than on it. One run was measured: the headroom is a
    # judgement, not a spread.
    "pass_threshold": 0.30,
    "dataset_label": "Aquarium Detection",
    "dataset_description": (
        "Roboflow Aquarium Combined v2 (CC BY 4.0), resized to 640 px. "
        "train.zip: 448 images with YOLO labels (class x_center y_center width "
        "height, normalised). test_images.zip: 190 images to detect on. "
        "classes.csv names the 7 classes. detection_metric.py is the "
        "leaderboard's parser and mAP50-95; sample_submission.csv shows the "
        "format and scores 0."
    ),
    # Deterministic scorer: the same CSV always yields the same mAP.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "map50_95", "label": "mAP50-95", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
        {"key": "map50", "label": "mAP50", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
        *[{"key": f"ap_{c}", "label": c, "source": "env",
           "agg": "mean", "format": "number", "precision": 3,
           "higher_is_better": True}
          for c in ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]],
        {"key": "n_detections", "label": "Detections", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0},
    ],
    "is_public_initial": False,
}
