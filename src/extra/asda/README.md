# ASDA comparator

This directory contains the complete ASDA comparator reported in the
manuscript (Section 3.6, Table 6 and Appendix E) and the code that produced
its 25 archived runs. ASDA was proposed in:

> Q. Gou and G. Cui, "Adversarial selective domain adaptation with feature
> cluster for skin cancer diagnosis," *Scientific Reports*, vol. 16, 223, 2026.
> <https://doi.org/10.1038/s41598-025-98293-5>

The code is an independent implementation from the article's equations and
Algorithm 1; no author implementation was used. The structured records of the
archived runs are in [`evidence/asda/`](../../../evidence/asda/).

## Protocol mapping

The implementation retains the article's source classification,
conditional-adversarial alignment, target feature clusters and signed
selective-entropy objective. The original report used ResNet-50, 200 epochs
and best-target-accuracy model selection. This comparator uses the shared
ResNet-18, 40 epochs (240 updates) and the average of the softmax
probabilities from epochs 36-40, which align model capacity and checkpoint
selection with the sonar comparison. No additional source-only warm-up or
target-label-guided configuration search was used. As stated in Appendix E.1,
the complete comparator was fixed before its target performance was evaluated,
and the later diagnostics did not change it.

The fixed protocol is the `PROTOCOL` record in [`common.py`](common.py)
(identifier `ASDA-SEL-01/A`). All 25 archived run configurations contain it
unchanged; its SHA-256 digest is recorded in
[`provenance.json`](../../../evidence/asda/provenance.json). The `provenance`
and `limits` fields of the record are short notes frozen with it. They name
the sources of the conventions (Algorithm 1 of the article, torchvision
RandAugment, the discriminator of the Microsoft GLS reference implementation,
the common protocol of the main comparison, and explicit adaptations) and the
three points that the article does not settle: the last-view entropy
interpretation, the software magnitude scale of RandAugment, and joint BN.

## Implementation conventions

The table repeats manuscript Table E.1 and adds where each convention is
implemented.

| Item | Implemented convention | Code |
|---|---|---|
| Backbone and input | ImageNet-pretrained ResNet-18 (torchvision `IMAGENET1K_V1`), 224x224 inputs; all backbone and classifier parameters trainable | `model.ASDAResNet18`, `common.init_path` |
| Target augmentation | Common augmented base view plus three independently generated RandAugment views; three operations per additional view | `data.BlindTargetDataset`, `data.target_cluster` |
| RandAugment convention | torchvision 0.24.1, 14-operation pool, magnitude index 2 on a 31-level scale, nearest interpolation and zero fill. Copies are rounded once to uint8; the base view keeps floating-point values | `data.TracedRandAugment`, `data.target_cluster` |
| Training-time BN | One joint forward batch: 64 source images, 64 target base views and 192 additional target views. Ordinary shared BN; evaluation uses the accumulated statistics without an extra AdaBN pass | `model.joint_batch`, `model.train_step` |
| Conditional alignment | Detached probabilities multiplied by features; separate source and target mean binary cross-entropy terms summed; gradient-reversal coefficient 1 | `model.conditional_input`, `model.domain_terms`, `model.Reverse` |
| Discriminator | Linear dimensions 1024-1024-1024-1, ReLU and dropout 0.5 | `model.Discriminator` |
| Source loss and coefficients | Common effective-number weighted cross-entropy (beta = 0.999); domain and signed-entropy coefficients both 1 | `model.source_weights`, `model.train_step` |
| Optimisation | SGD with momentum 0.9, Nesterov and weight decay 5e-4; initial learning rates 0.001 for the backbone and 0.01 for the classifier and discriminator; common inverse decay over 240 updates | `model.optimizer_scheduler` |
| Evaluation endpoint | Average softmax probabilities from epochs 36-40; compute fold metrics, then average folds within each seed. Five seeds are the reporting units | `artifacts.average_five_predictions`, `evaluate_asda.seed_summary` |

The common base augmentation, class weights, learning rates and decay are
those of the CDAN trainer in [`src/run2_s5_cdan/`](../../run2_s5_cdan/);
`tests/test_asda.py` checks this directly.

The target base view already receives the common stochastic augmentation. In
the article's notation, N = 3 is the number of augmented views in its Eq. (2);
its experimental settings also use this value for the number of augmentation
operations, while M = 2 controls augmentation severity. These values do not
uniquely determine the software magnitude scale. This implementation uses
three views with three operations per view and magnitude index 2 on the
31-level scale. Detaching the conditioning probabilities routes the domain
objective's adversarial gradient through the features without a direct
classifier gradient through the conditioning path. Separate domain means keep
the larger number of target views from implicitly determining the relative
domain-loss weight.

A cluster is reliable when at least two of its three augmented predictions
agree with the base prediction. For target base image i, let r_i = 1 for a
reliable cluster and r_i = 0 otherwise. With B_t = 64 target base images
(clusters) per batch, the signed entropy contribution
(`model.signed_entropy`) is

```text
L_H = (1 / B_t) * sum_i (2 r_i - 1) * H(p_i^(3)),    H(p) = -sum_c p_c log p_c,
```

where p_i^(3) is the class-probability vector of image i in the third (last
generated) augmented view. Reliable clusters minimise entropy and unreliable
clusters maximise it. Algorithm 1 of the article uses the last augmented
prediction in both branches, whereas its prose also refers to a consistent
augmented version. This implementation uses the final generated view: a base
prediction A with augmented predictions [A, A, B] forms a reliable cluster
whose entropy term acts on B. This interpretation is used throughout and does
not assert how unreported author code would resolve the ambiguity.

## Target-label boundary

Training reads the 553 target images only through the label-free loader of
[`src/_klsg_blind.py`](../../_klsg_blind.py), which returns a poison sentinel in
place of each label; `data.BlindTargetDataset` rejects any other item.
`train_asda.py` does not use target labels or score the target.

`evaluate_asda.py` first validates the five saved checkpoints of a run,
computes and saves the target probabilities, and only then reads the 553
target labels through the authorization gate of `src/_klsg_blind.py`. The gate
requires `--authorize-klsg-labels` with a `--reason` and writes an access log
into the run's evaluation directory. In the archived execution, the training
stage for all 25 runs ended at 2026-09-17T04:25:01Z and the first target label
was read at 2026-09-17T04:26:21Z (`provenance.json` and the per-run records).

## Running

The comparator needs a CUDA GPU; the trainer does not fall back to the CPU. It
uses the data layout described in the [main README](../../../README.md):
source images under `data/raw/ai4simulate/`, the complete KLSG-II target under
`data/raw/SeabedObjects/`, and the split files in `data/split/`. On first use,
`common.init_path()` downloads the torchvision ImageNet ResNet-18 weights into
the torch hub cache and verifies their SHA-256 digest against the one recorded
for the archived runs. Versions other than the recorded Python 3.11.15,
PyTorch 2.9.1+cu128 and torchvision 0.24.1+cu128 produce a warning, not an
error; bit-identical results are not expected on other software or hardware.
Deterministic algorithms are enabled, and `CUBLAS_WORKSPACE_CONFIG` defaults to
`:4096:8`.

One run, from the repository root:

```text
python src/extra/asda/train_asda.py --seed 0 --fold 0
python src/extra/asda/evaluate_asda.py --seed 0 --fold 0 --authorize-klsg-labels --reason "post-freeze final scoring"
```

After all 25 runs have been evaluated, the five-seed summary and the paired
contrasts with source-only (#1) and source-only+AdaBN (#2) are written by:

```text
python src/extra/asda/evaluate_asda.py --aggregate
```

`run_matrix.py` prints the complete plan of 25 trainings, 25 evaluations and
one aggregation and runs nothing unless `--execute` is given; stages that read
target labels additionally need the authorization flag and a reason:

```text
python src/extra/asda/run_matrix.py
python src/extra/asda/run_matrix.py --stage train --execute
python src/extra/asda/run_matrix.py --stage evaluate --execute --authorize-klsg-labels --reason "post-freeze final scoring"
python src/extra/asda/run_matrix.py --stage aggregate --execute
```

A completed run is validated rather than retrained. An interrupted attempt is
kept together with an `INTERRUPTED.json` record, and `--restart-incomplete`
restarts the whole run from its fixed seed in a new attempt directory.

Outputs are written under `results/extra/asda/`, which is git-ignored:

```text
results/extra/asda/
  runs/asda_s{seed}_f{fold}/
    TRAINING_COMPLETE.json        completion marker with checkpoint and artifact digests
    attempt_001/
      config.json                 seed, fold, protocol, input and source digests, environment
      steps.jsonl                 one record per update
      cost.json                   update counts, forward multiply-accumulate counts, timing, CUDA peaks
      asda_epoch036.pth ... asda_epoch040.pth
      evaluation/
        epoch036.npz ... epoch040.npz   target probabilities of each checkpoint
        predictions.npz                 five-checkpoint mean probabilities
        metrics.json                    fold metrics
        bundle/                         arrays for diagnostics, including target labels
        _LABEL_ACCESS_LOG.txt
        evaluation_cost.json
        EVALUATION_COMPLETE.json
  summary/summary.json            five-seed summary and paired contrasts
```

## Archived evidence

[`evidence/asda/`](../../../evidence/asda/) holds the structured records of
the 25 archived runs:

| File | Contents |
|---|---|
| `per_run/asda_s{seed}_f{fold}.json` | Fold metrics and confusion matrix, predicted class counts, training and prediction times, CUDA peaks, forward multiply-accumulate counts, first label-read time, and the SHA-256 digests of the configuration, the five checkpoints and the other run artifacts |
| `reference_arms.json` | Seed-level values of source-only (#1) and source-only+AdaBN (#2), copied from the frozen aggregate of the BN-focused comparison |
| `summary.json` | The summary that `evaluate_asda.py --aggregate` writes, recomputed from the archived records with the public code; `reference_sha256` is the digest of `reference_arms.json`, whose `source_aggregate_sha256` identifies the executed reference aggregate |
| `summary.md` | The manuscript values recomputed from these records, by table |
| `provenance.json` | Frozen protocol, SHA-256 digests of the executed source files and inputs, initialisation weights, environment and stage times |
| `SHA256SUMS.txt` | Digests of the files above |

`tests/test_asda.py` recomputes the summary from `per_run/`, recomputes the
paired contrasts from `reference_arms.json`, and compares the results with the
values printed in the manuscript.

## Relation to the executed code

The archived runs were executed from a working copy of this code.
`data.py`, `train_asda.py` and `artifacts.py` are byte-identical to the
executed files. `common.py`, `model.py`, `evaluate_asda.py` and
`run_matrix.py` do not change how any reported value is computed. They differ
in local paths and the configuration fields that recorded local files, the
software-version check (now a warning), the loading of the ImageNet
initialisation (now downloaded and verified by SHA-256), the scoring import,
relative paths in output records, and the reference file; `provenance.json`
describes the changes to each file. `scoring.py` contains the executed
scoring functions with unchanged code; the executed evaluator imported them
from helper modules of the working copy. The executed aggregation also
computed contrasts with #5 and #7, which the manuscript does not report; this
code computes only the two reported contrasts. `provenance.json` lists the digests of the executed files
and states, for each of them, whether it is public and how the public file
relates to it. This includes `preflight.py` and the two scoring helper modules,
which are not public, and `src/_klsg_blind.py` and
`src/run2_s5_cdan/train_cdan_sgd.py`, whose public versions differ from the
executed working copies.

## Not included

- Model checkpoints, per-checkpoint and averaged target probabilities,
  features, and the other arrays of the evaluation bundle.
- The per-image target labels and target image paths of the evaluation
  bundle (`bundle/y_true.npy`, `bundle/tgt_paths.txt`), and any image data.
  The KLSG-II split metadata in `data/split/`, including the class labels
  that the authorized evaluation reads, has been public since v1.0.0.
- The per-update logs (`steps.jsonl`) and the label-access logs; each per-run
  record lists their SHA-256 digests.
- The code of the post-hoc diagnostics in Appendices E.3 and E.4: the matched
  four-condition study, which trained 100 further condition runs (Table E.4),
  the frozen-output gate diagnostic computed from their saved predictions
  (Table E.5), and the same-input BN probes described in Appendix E.4. These
  diagnostics did not change the comparator's configuration or checkpoints.
- `preflight.py`, the pre-run acceptance script listed among the executed
  source digests. Several of its checks need the local images and earlier
  checkpoints; its checks that need neither are reproduced on the CPU in
  `tests/test_asda.py`, with a small stand-in network in place of ResNet-18
  for the training step.

## Tests

```text
python -m unittest tests.test_asda -v
```

The tests need no data, network access or GPU.
