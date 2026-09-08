"""Opaque, salted row ids for the file_v1 datasets.

Every dataset in this directory is a split of a **public** source — openml, UCI,
sklearn — and `prepare_data.py` is committed to a public repository. So the
split is reproducible by anyone who reads it: same fetch, same
`train_test_split(..., random_state=SEED)`, same rows in the same order. While
the shipped ids run `te_00000, te_00001, …` in exactly that order, the held-out
target is a three-line copy and no modelling is involved:

    _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pd.DataFrame({"id": X_test["id"], "prediction": y_test.values})   # R2 = 1.0

The fix is to break the correspondence between a row's position in the
reproducible split and the id it is published under, using a permutation that
is **not in the repository**. `MLARENA_ID_SALT` supplies it. Given the same
salt the assignment is byte-for-byte reproducible; without it the build stops
rather than falling back to a public default.

Keep the salt outside the repo. It is what lets you rebuild an already-published
dataset identically — regenerate under a different salt and every id changes,
which invalidates the ids students have already downloaded and every submission
written against them.

**What this does not fix.** A student who identifies the source dataset can
still join `X_test` back to it on the feature columns and read the target off;
on both Session 2 datasets that join is exact and unambiguous for 100% of test
rows. Nothing that leaves the features intact can prevent it. This defends
against replaying the split, which is the cheap attack — not against
reconstructing the source, which is the expensive one.
"""
import hashlib
import os

import numpy as np

SALT_ENV = "MLARENA_ID_SALT"
_ID_BYTES = 6  # 12 hex chars; uniqueness is asserted below regardless


def _salt() -> bytes:
    salt = os.environ.get(SALT_ENV)
    if not salt:
        raise SystemExit(
            f"{SALT_ENV} is not set.\n"
            f"The row ids are salted so the published order cannot be derived "
            f"from the split, which is reproducible from this file. Export the "
            f"salt used for the live datasets — losing it means rebuilding "
            f"under a new one, which changes every id.\n"
            f"  export {SALT_ENV}=$(openssl rand -hex 32)   # a new dataset only"
        )
    return salt.encode()


def _digest(*parts: str, size: int) -> bytes:
    return hashlib.blake2b(
        "/".join(parts).encode(), key=_salt(), digest_size=size
    ).digest()


def shuffle_and_label(X, y, *, dataset: str, split: str, prefix: str,
                      shuffle: bool = True):
    """Shuffle a split into a salt-derived order and give it opaque ids.

    `dataset` and `split` are mixed into the key so no two splits share a
    permutation. Returns `(X, y)` with a fresh contiguous index, `X` carrying
    the ids in column 0 and `y` aligned to them.

    `shuffle=False` labels the rows without reordering them, for a **training**
    split whose row order is itself information the student needs — a time
    series ordered past-to-future, where the honest way to carve a validation
    set is to take the last rows rather than random ones. There is nothing to
    protect there: the training targets ship in `y_train.csv` anyway, so the
    permutation was defending an asset that was never secret. Never pass it for
    a test split, whose held-back targets are exactly what the shuffle defends
    (see the module docstring).
    """
    n = len(X)
    if n != len(y):
        raise SystemExit(f"{dataset}/{split}: X has {n} rows, y has {len(y)}")

    if shuffle:
        seed = int.from_bytes(_digest(dataset, split, "order", size=8), "big")
        order = np.random.default_rng(seed).permutation(n)
    else:
        order = np.arange(n)

    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)

    ids = [f"{prefix}_{_digest(dataset, split, str(i), size=_ID_BYTES).hex()}"
           for i in range(n)]
    if len(set(ids)) != n:
        raise SystemExit(f"{dataset}/{split}: id collision over {n} rows")

    X.insert(0, "id", ids)
    return X, y
