# MNIST Warm-up — Session 4

Classify 5,000 handwritten digits with the network you trained in **Lab 4**.

This is the dry run for the project. Lab 4 Part D is explicit about what it is
for: *getting on the board matters; your position does not.* The thing being
tested is that your training pipeline ends in a file the platform accepts —
while that is still cheap to get wrong.

## The data

| file | rows | contents |
|---|---|---|
| `X_test.csv` | 5,000 | `id` + `p0 … p783` — predict these |
| `sample_train.csv` | 2,000 | `id,label` + `p0 … p783` — a format reference |

**The pixel convention matters.** `p0 … p783` are the 28×28 image flattened
row-major as `uint8` in `0–255` — exactly what `torchvision.datasets.MNIST`
gives you *before* `ToTensor()` divides by 255. Train on the full torchvision
train split as Lab 4 Part C describes; `sample_train.csv` is here so you can
assert your loader agrees with ours before you trust a submission:

```python
row = sample.iloc[0]
img = row[[f"p{i}" for i in range(784)]].to_numpy(dtype="uint8").reshape(28, 28)
# img should look like row["label"]. If it looks transposed, you have found
# the bug this file exists to catch.
```

Remember the normalisation your model was trained with. A model fed raw `0–255`
when it expected `(x - 0.1307) / 0.3081` will not fail — it will just be
confidently wrong, which is worse.

## What you submit

`submission.csv` — one row per test id, in any order:

```csv
id,label
te_00000,7
te_00001,2
```

`label` is the predicted digit, `0–9` — the class, not a probability and not a
one-hot row. Every id in `X_test.csv` must appear exactly once; anything else is
rejected with a message naming the line.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(competition_id=<id>, files=["submission.csv"])
print(client.leaderboard(<id>).head())
```

## Scoring

**Accuracy.** Macro-F1 and the weakest digit's F1 are also shown — those are the
ones that notice when a model has quietly stopped predicting a class, which
plain accuracy on ten roughly balanced classes will hide from you.

Multinomial logistic regression on raw pixels scores **91.3%**. An MLP that is
wired up correctly clears 97%. If you are below the logistic-regression line,
the problem is not your architecture — it is the normalisation, a `softmax`
before `CrossEntropyLoss`, or a missing `model.eval()`, and the one-batch
overfit test from Part B would have told you which.

## What this measures, honestly

The images come from public MNIST, so the labels exist somewhere on the
internet. This is a plumbing check by design and it is scored as one. The
project competitions are not, and neither is anything in *MS2A - Machine
Learning Practice* — which is why doing this one properly now is worth the
forty-five minutes.
