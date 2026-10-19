# Step 1 — Train and Save a Model in Colab

The workshop serves one model from start to finish: a penguin species
classifier. This step trains it in Colab and saves it to one file,
`model.joblib`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s10-penguins-train.ipynb)

<!-- notes: 15 minutes. Everyone runs "Runtime > Run all" in the first two
minutes; the walk-through happens while it runs. Numbers below come from the
notebook executed on Colab's runtime of 2026-10-19 (Python 3.13, scikit-learn
1.6.1, joblib 1.6.0, pandas 2.2.3). The fit cell prints an
`OptimizeWarning: Unknown solver options: iprint`: it comes from scikit-learn
1.6.1 with a recent scipy and is harmless. Where students get stuck: the
browser blocks the download pop-up of the last cell (allow pop-ups for
colab.research.google.com, or use the Files panel > model.joblib > Download),
and Safari sometimes saves it as `model.joblib.txt` (rename it). Make them
write down the versions cell: Step 2 pins scikit-learn to it. -->

---

## The data: Palmer penguins

![Chinstrap, Gentoo and Adélie penguins](assets/deploy/penguins.png)

*Artwork by @allison_horst.*

- 344 penguins of 3 species, measured on 3 islands of Antarctica, 2007–2009
  (CC0, [palmerpenguins](https://github.com/allisonhorst/palmerpenguins)).
- 2 rows have no measurement and are dropped: 342 penguins remain, 151
  Adelie, 123 Gentoo, 68 Chinstrap.

---

## Four measurements

![Bill length and bill depth on a penguin's bill](assets/deploy/culmen-depth.png)

*Artwork by @allison_horst.*

The model reads four numbers: `bill_length_mm`, `bill_depth_mm`,
`flipper_length_mm`, `body_mass_g`. These four names are, from Step 2 on,
the fields of the API's request.

---

## What the notebook does

look → split → baseline → pipeline → score → save

| Quantity | Value |
|---|---|
| train / test split (25 %, stratified, seed 0) | 256 / 86 |
| baseline: always `Adelie` | 0.4419 |
| scaler + logistic regression, test accuracy | 0.9884 |

The three lines that produce the file:

```python
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
joblib.dump(model, "model.joblib")
```

---

## Why a pipeline

- The `StandardScaler` learns a mean and a standard deviation per feature on
  the training set. The pipeline stores them **inside** `model.joblib`, next to
  the regression weights.
- The API therefore sends raw millimetres and grams; the loaded pipeline
  scales them exactly as during training.
- The file is 2,033 bytes: means, deviations, weights, class names. It does not
  contain scikit-learn itself.
- `model.joblib` is a pickle. Loading a pickle runs code: a model file is
  loaded only from a trusted source.

---

## The versions that wrote the file

The notebook's last cells print (Colab, October 2026):

```text
python       3.13.16
scikit-learn 1.6.1
joblib       1.6.0
pandas       2.2.3
```

A pickle is read back reliably by the **same** scikit-learn version. Step 2
pins `scikit-learn==1.6.1` in the project: these numbers are the
reproducibility contract between Colab, the laptop and the image.

**Checkpoint:** `ls -l ~/Downloads/model.joblib` → a file of 2,033 bytes.

---

## Check yourself

1. Why is the baseline 0.4419? **It is the share of Adelie, the most frequent
   species, in the test set: a model is useful only above it.**
2. The API receives `bill_length_mm = 39.1`, unscaled. Where does the scaling
   happen? **Inside the loaded pipeline: the scaler's mean and deviation are
   saved in `model.joblib`.**
3. Why write down the scikit-learn version printed by Colab? **The project of
   Step 2 must load the pickle with the same version that wrote it.**
