# Aquarium Detection — Session 6

Find every animal in the picture: where it is, and what it is.

Each image comes from one of two public aquariums and holds anything from one
to several dozen animals of **7 classes**: fish, jellyfish, penguin, puffin,
shark, starfish, stingray. For every test image you submit a list of
**bounding boxes**, each with a class and a confidence. The leaderboard ranks
you on **mAP50-95**, the box metric Ultralytics prints during training.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s6-aquarium-detection.ipynb)

**The starter notebook** downloads the files, draws the labelled boxes,
fine-tunes a COCO-pretrained **YOLOv8n** for 30 epochs, scores it on a
held-out slice of the training images with the leaderboard's own metric, and
submits. Use a GPU runtime (Runtime → Change runtime type → T4 GPU): training
takes about 5–10 minutes.

## The files

| file | contents |
|---|---|
| `train.zip` | 448 images (`images/<id>.jpg`) and their labels (`labels/<id>.txt`) |
| `test_images.zip` | 190 images to detect on, no labels |
| `classes.csv` | `class_id,name` for the 7 classes |
| `sample_submission.csv` | every test id with no detection: the format, scoring 0 |
| `detection_metric.py` | the parser and the metric the leaderboard runs |

Images are 640 px on their long side. A label file has one line per box in
the **YOLO format**, every coordinate normalised by the image's width or height:

```text
class_id x_center y_center width height
0 0.412 0.530 0.120 0.085
```

## The submission

`submission.csv`, **one row per test image**, all 190 of them:

```text
image_id,prediction_string
te_016ee383598d,0 0.91 0.412 0.530 0.120 0.085 4 0.33 0.702 0.201 0.310 0.144
te_0185dd1595bf,
```

`prediction_string` lists the detections, six numbers each: `class_id
confidence x_center y_center width height`, in the same normalised YOLO
coordinates as the labels (Ultralytics gives them as `boxes.cls`,
`boxes.conf` and `boxes.xywhn`). An empty string means no detection. At most
300 detections per image; confidences in [0, 1]. A file that breaks any rule
is rejected with a message naming the line, and never scored.

`detection_metric.to_prediction_string(boxes)` writes the cell, and
`detection_metric.score_file(submission, ground_truth)` scores a file in the
same layout, so your validation number is computed by the leaderboard's code.

## The score

**mAP50-95** on the 190 private images (1,491 boxes). For each class,
detections are sorted by confidence and matched to ground-truth boxes of the
same class in the same image, greedily by **IoU**. At an IoU threshold *t*, a
match is a true positive; everything else is a false positive or a miss. AP is
the area under the precision-recall curve; the score averages it over the ten
thresholds 0.50, 0.55, …, 0.95 and over the 7 classes.

- A box in roughly the right place counts at 0.50 and not at 0.90: mAP50-95
  rewards **tight** boxes, mAP50 (shown beside it) only rough ones.
- Every class weighs the same: the rare **puffins** and **stingrays** count
  as much as the hundreds of fish. The per-class AP columns show where you lose.
- Low-confidence detections cost nothing at the top of the ranking and add
  recall at the bottom: predict with a **low confidence threshold** (0.001,
  the Ultralytics validation default), not the 0.25 used to draw pictures.
- The ceiling is 0.995, not 1: the 101-point sampling of the curve, as in
  Ultralytics.

The **benchmark** is the starter notebook's model, YOLOv8n fine-tuned for 30
epochs. Beat it.

## Honest limits

The images and labels are public (Roboflow *Aquarium Combined*, CC BY 4.0),
so the test labels can be found. Ids are renamed and images resized to make
that a chore rather than a lookup; doing it is not what is assessed here.

*Dataset: [Roboflow Aquarium Combined v2](https://universe.roboflow.com/brad-dwyer/aquarium-combined),
Henry Doorly Zoo (Omaha) and National Aquarium (Baltimore), CC BY 4.0.*
