# Forest Cover Type — Session 4

Predict which of seven tree species dominates a 30 × 30 metre patch of the
Roosevelt National Forest in Colorado, from 54 cartographic features. 14,000
training patches, 3,500 to predict, seven balanced classes.

This is the **classification** challenge of Session 4, and unlike its sibling
it comes with **no worked code**. You have just read the California-housing
notebook, which builds a PyTorch regressor from a bare tensor up to a training
loop. This is the same machinery pointed at a class label, and you write it.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s4-forest-cover.ipynb)

The notebook is the protocol in English with empty cells under it. Three things
change from the regression case and the notebook makes you find all three:
the last layer's width, the loss function, and how you turn the network's
output back into an answer.

## The data

| file | rows | contents |
|---|---|---|
| `X.csv` | 14,000 | `id` + the 54 features |
| `y.csv` | 14,000 | `id,prediction` — the cover type, 1–7 |
| `X_submission.csv` | 3,500 | `id` + the same 54 features |

`X.csv` and `y.csv` are the labelled data — fit on them, and carve your own
validation split out of them. `X_submission.csv` is what the leaderboard
scores; its labels are held back, so it cannot serve as a validation set.

A balanced subsample of the UCI Covertype dataset (scikit-learn's
`fetch_covtype`): **2,500 rows of each of the seven cover types**, split 80/20
stratified at `random_state=42`. No missing values.

| block | columns | what they are |
|---|---|---|
| terrain | `elevation`, `aspect`, `slope` | metres, compass bearing 0–360, degrees |
| distances | `h_dist_hydrology`, `v_dist_hydrology`, `h_dist_roadways`, `h_dist_fire_points` | metres to water, roads, fire ignition points |
| light | `hillshade_9am`, `hillshade_noon`, `hillshade_3pm` | shade index 0–255 at three times of day |
| area | `wilderness_area_1` … `_4` | four 0/1 indicators, exactly one set per row |
| soil | `soil_type_1` … `_40` | forty 0/1 indicators, exactly one set per row |

The target:

| code | cover type | code | cover type |
|---|---|---|---|
| 1 | Spruce/Fir | 5 | Aspen |
| 2 | Lodgepole Pine | 6 | Douglas-fir |
| 3 | Ponderosa Pine | 7 | Krummholz |
| 4 | Cottonwood/Willow | | |

**Why balanced.** The full dataset is 581,012 rows, 49% lodgepole pine and
0.47% cottonwood. On that distribution accuracy stops meaning anything — a
model that never predicts cottonwood loses half a percent and looks fine.
Sampling 2,500 of each makes chance exactly 1/7 = **0.143**, makes accuracy
equal to balanced accuracy, and makes macro-F1 track it. One number on the
leaderboard is then honest. Cottonwood has 2,747 rows in the source, which is
what caps the per-class count.

**Note the 1-based coding**, because it will cost you twenty minutes otherwise.
`torch.nn.CrossEntropyLoss` expects class *indices* starting at 0. Your labels
start at 1. So it is `y - 1` on the way into the loss and `argmax(dim=1) + 1`
on the way back out. Forget the first and you get an `IndexError` about a
target out of bounds — loud, and easy. Forget the second and you submit a
column shifted by one, which is not loud at all: it is simply an accuracy near
zero with no error anywhere. The scorer rejects a `0` with a message saying so.

## What you submit

`submission.csv` — one row per id in `X_submission.csv`, in any order:

```csv
id,prediction
te_00000,2
te_00001,7
```

`prediction` is the integer cover type, 1–7. Not a probability, not a row of
logits, not a one-hot vector. Every id in `X_submission.csv` must appear exactly
once; a missing id, an unknown id, a duplicate or a value outside 1–7 is
rejected with a message naming the line, and scores 0.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=<id>, files=["submission.csv"])
```

## Scoring

**Accuracy**, on the 3,500 held-out patches — meaningful because the classes
are balanced. **Macro-F1** rides along, and so does the **worst class's F1**,
which is the column that notices when a model has quietly stopped predicting a
cover type altogether. An under-trained 7-way head does exactly that: it parks
on the two or three easy classes and ignores the rest. Accuracy alone can look
respectable while that happens; the worst-class column cannot.

## The point of this challenge

Every row measured on this exact split.

Every MLP row uses the protocol section 6 of the notebook asks for: 60 epochs of
Adam at `lr=1e-3`, batch size 128, best-validation checkpoint restored.

| model | test accuracy | macro-F1 |
|---|---|---|
| predict the most common class | 0.143 | 0.036 |
| **LogisticRegression on standardised features** — the benchmark | **0.696** | 0.692 |
| MLP 54-128-64-7, **inputs not standardised**, 60 epochs | 0.737 | 0.728 |
| MLP 54-128-64-7, standardised, 20 epochs | 0.770 | 0.766 |
| **MLP 54-128-64-7, standardised, Adam, 60 epochs** | **0.814** | 0.812 |
| MLP 54-256-128-7, standardised, Adam, 60 epochs | 0.815 | 0.811 |

The shape of the story is the same as the regression half: the linear model has
a ceiling at 0.696, and a small MLP clears it by twelve points. The
unstandardised row is less catastrophic here than on California housing —
44 of the 54 columns are already 0/1 indicators, so most of the matrix is
already on one scale — but it still costs about eight points against the
identical network, and `elevation` in metres against `slope` in degrees is why.

The last row is worth reading as a warning rather than an invitation: four times
the parameters buys 0.001. You are at the point where the data, not the model,
is the constraint.

Your target is the benchmark: **beat 0.696.** Getting to 0.81 is a matter of
standardising, training long enough, and watching a validation split rather
than a training loss. Across seeds 0, 1 and 2 that recipe lands on 0.814, 0.810
and 0.813, so a much lower number means something is wrong rather than unlucky.
Getting past 0.83 is a different exercise and not the point of this session.

Submissions are scored once and deterministically. Choose your model on a
validation split you hold out yourself; use the leaderboard to confirm, not to
search.
