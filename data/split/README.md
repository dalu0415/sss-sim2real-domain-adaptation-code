# Split metadata and target-label boundary

This directory freezes the split and target-manifest metadata used by the
paper:

- `splits.csv`: the deterministic five-fold source and supervised-ceiling
  assignments;
- `splits_manifest.json`: split seed, class counts, and the SHA-256 digest of
  `splits.csv`;
- `klsg_unlabeled_manifest.txt`: the exact 553-image target order used by all
  unsupervised adaptation arms;
- `klsg_churn100_manifest.txt`: the fixed class-blind 100-image subset used
  only for label-free churn diagnostics;
- `klsg_churn100_manifest.json`: the seed, sampled row indices, and hashes for
  that fixed subset.

`make_splits.py` uses
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. It reads target
class-directory names only to construct the supervised-ceiling folds. Target
labels are not available to UDA losses, target-weight estimation, model
selection, or early stopping. The UDA loaders return a poison sentinel in the
label position and reject direct KLSG-label reads.

The KLSG-II rows in `splits.csv` contain relative filenames and class labels
derived from the upstream class-directory organization so the supervised
ceiling split can be reproduced. These are split metadata only; no KLSG-II
image bytes are redistributed.

The permitted target-label uses are:

1. deterministic stratification for the supervised-ceiling folds;
2. supervised-ceiling training and held-out scoring through the explicit
   authorization gate;
3. final frozen-model scoring in the separate evaluation stage.

Expected image layout:

```text
data/raw/ai4simulate/{airplane,ship}/
data/raw/SeabedObjects/{airplane,ship}/
```

The SeabedObjects-KLSG-II provenance repository is:
<https://github.com/HHUCzCz/-SeabedObjects-KLSG--II>. Its current contents no
longer include the complete paper dataset. Users must obtain the full dataset
lawfully from its authors or another authorized source; its images are not
redistributed here.
