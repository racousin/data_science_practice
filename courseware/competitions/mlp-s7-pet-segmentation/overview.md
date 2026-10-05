# Pet Segmentation — Session 7

Paint the cat or the dog: for every pixel of the photo, is it pet or background?

Each image is a photo of one of 37 cat and dog breeds, resized to
**128 × 128**. For every test image you submit a **mask**: the set of pixels
you think are the pet. The leaderboard ranks you on **mean IoU** over the two
classes, background and pet.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s7-pet-segmentation.ipynb)

**The starter notebook** downloads the files, shows the images and their
masks, trains a **small U-Net written from scratch** and submits it, then
**fine-tunes a U-Net with an ImageNet-pretrained ResNet-18 encoder**
(`segmentation_models_pytorch`) and submits again. Use a GPU runtime
(Runtime → Change runtime type → T4 GPU): the whole notebook takes about 10
minutes.

## The files

| file | contents |
|---|---|
| `train.zip` | 5,849 photos (`images/<id>.jpg`) and their masks (`masks/<id>.png`) |
| `test_images.zip` | 1,500 photos to segment, no masks |
| `sample_submission.csv` | every test id with an empty mask: the format |
| `seg_metric.py` | the format helpers and the metric the leaderboard runs |

A mask is a 128 × 128 PNG with one value per pixel:

| value | meaning |
|---|---|
| 0 | background |
| 1 | pet |
| 2 | border: the thin band around the pet the annotators left undecided — **not scored** |

## The submission

`submission.csv`, **one row per test image**, all 1,500 of them:

```text
image_id,mask_rle
te_016ee383598d,1153 9 1280 13 1407 16
te_0185dd1595bf,
```

`mask_rle` is the **run-length encoding** of your pet pixels: flatten the
128 × 128 mask row by row (`mask.flatten()`), then write `start length` pairs,
starts counted from 1. An empty string means no pet pixel.
`seg_metric.rle_encode(mask)` writes it and `seg_metric.rle_decode` reads it
back; a file that breaks the format is rejected with a message naming the
line, and never scored.

## The score

For each class *c* (background, pet), over every scored pixel of the 1,500
test images together:

$$\text{IoU}(c) = \frac{|\text{pred} = c \;\cap\; \text{true} = c|}{|\text{pred} = c \;\cup\; \text{true} = c|}
\qquad \text{mIoU} = \frac{\text{IoU(background)} + \text{IoU(pet)}}{2}$$

- Border pixels (value 2) count for neither class: draw the boundary anywhere
  inside the band.
- Predicting *all background* scores about **0.33**, not 0: IoU(background)
  is then the background's share of the pixels.
- The pet IoU, the background IoU, the pet Dice and the pixel accuracy are
  shown beside the score. `seg_metric.score_masks(true, pred)` computes all
  of them on your validation masks, with the leaderboard's code.

The **benchmark** is the starter notebook's first model, the small U-Net
trained from scratch for 15 epochs. The fine-tuned U-Net of the notebook's
second part beats it. Beat both.

## Honest limits

The photos and masks are public (Oxford-IIIT Pet, CC BY-SA 4.0), so the test
masks can be found. Ids are renamed, the split reshuffled and the images
resized to make that a chore rather than a lookup; doing it is not what is
assessed here.

*Dataset: [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/),
O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs", CVPR 2012,
CC BY-SA 4.0. Images and masks are resized here.*
