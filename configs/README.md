# Training and experiment configuration

`experiment_matrix.yaml` is a descriptive, frozen record of the configurations
used in the paper; it is not loaded automatically by the trainers. The
CLI-exposed values are passed by the command templates below, while the other
locked settings are explicit constants in the corresponding scripts. The
historical internal batch drivers are intentionally not published because they
contained machine-specific paths and destructive cleanup operations.

The public scripts infer the repository root from their own location. They do
not contain an author-machine fallback. Place the data under the repository
layout below:

```text
data/
  raw/
    ai4simulate/
      airplane/
      ship/
    SeabedObjects/
      airplane/
      ship/
  split/
    splits.csv
    splits_manifest.json
    klsg_unlabeled_manifest.txt
    klsg_churn100_manifest.txt
```

This repository does not distribute the semi-synthetic source image files.
Users who lawfully obtain or construct source images should place their
`airplane` and `ship` directories under `data/raw/ai4simulate/`. The
SeabedObjects-KLSG-II provenance repository no longer contains the complete
paper dataset. Users who lawfully obtain the full target dataset from its
authors or another authorized source should place its two class directories
under `data/raw/SeabedObjects/`. No third-party target images are redistributed
in this repository.

## Per-run command templates

Run these commands from the repository root. Run each `{seed}` in `0..4` and
each `{fold}` in `0..4`. The source and UDA trainers write the curve, metrics,
churn trace, and checkpoints into the same seed/fold-specific result directory.
The supervised ceiling intentionally has no churn trace and keeps checkpoints
and metrics in separate seed/fold-specific directories. Every trainer refuses
to reuse a non-empty run directory, preventing stale checkpoints from entering
last-K selection.

```text
python src/run1_s4_baseline_ceiling/train_source_only_sgd.py --epochs 40 --seed {seed} --fold {fold}

python src/run2_s5_cdan/train_cdan_sgd.py --epochs 40 --seed {seed} --fold {fold}
python src/run2_s5_cdan/train_cdan_sgd.py --epochs 40 --seed {seed} --fold {fold} --entropy

python src/run3_s5_iwcdan/train_iwcdan_sgd.py --epochs 40 --seed {seed} --fold {fold}

python src/run4_s5_iwcdan_frozen/train_iwcdan_frozen_sgd.py --epochs 40 --seed {seed} --fold {fold} --config-name main --lr-mult 1.0 --trade-off 1.0
python src/run4_s5_iwcdan_frozen/train_iwcdan_frozen_sgd.py --epochs 40 --seed {seed} --fold {fold} --config-name diag --lr-mult 0.1 --trade-off 0.25

python src/run5_s5_iwcdan_dsbn/train_iwcdan_dsbn_sgd.py --epochs 40 --seed {seed} --fold {fold}

python src/extra/dann/train_dann_sgd.py --epochs 40 --seed {seed} --fold {fold}
python src/extra/mcc/train_mcc_sgd.py --epochs 40 --seed {seed} --fold {fold} --temperature 2.5 --mu 1.0
python src/extra/mdd/train_mdd_sgd.py --epochs 40 --seed {seed} --fold {fold} --margin 4.0 --trade-off 1.0
```

The frozen-BN diagnostic additionally used seed `5` under the preregistered
variance rule. Primary paired comparisons use the common seeds `0..4`.

The supervised ceiling is deliberately gated because it is the only training
arm that consumes target labels:

```text
python src/run1_s4_baseline_ceiling/train_ceiling_sgd.py --epochs 40 --seed {seed} --fold {fold} --authorize-klsg-labels --reason "supervised ceiling"
```

## Last-K and AdaBN

All formal models use epochs 36–40 (`K=5`). Last-K means applying softmax to
each checkpoint and averaging probabilities; it is not logit or parameter
averaging.

The five `eval_lastk.py` thin entries implement that operation. For the frozen
arm, also pass `--label main` or `--label diag` so the correct result directory
is selected.

```text
python src/run1_s4_baseline_ceiling/eval_lastk.py --seed {seed} --fold {fold} --k 5
python src/run2_s5_cdan/eval_lastk.py --variant cdan --seed {seed} --fold {fold} --k 5
python src/run2_s5_cdan/eval_lastk.py --variant cdan-e --seed {seed} --fold {fold} --k 5
python src/run3_s5_iwcdan/eval_lastk.py --seed {seed} --fold {fold} --k 5
python src/run4_s5_iwcdan_frozen/eval_lastk.py --label main --seed {seed} --fold {fold} --k 5
python src/run4_s5_iwcdan_frozen/eval_lastk.py --label diag --seed {seed} --fold {fold} --k 5
python src/run5_s5_iwcdan_dsbn/eval_lastk.py --seed {seed} --fold {fold} --k 5
```

AdaBN is available for source-only, CDAN, and IW-CDAN. Pass the five checkpoint
paths explicitly with `--ckpt`. Each checkpoint is independently loaded,
its BN running statistics are reset, and one cumulative-average pass over all
553 unlabeled target images is performed before probabilities are averaged.

The following templates expand the five AdaBN checkpoint arguments explicitly;
replace `{seed}` and `{fold}` before running:

```text
python src/run1_s4_baseline_ceiling/eval_adabn.py --seed {seed} --fold {fold} --passes 1 --ckpt results/run1_s4_baseline_ceiling/sgd_full_s{seed}_f{fold}/sgd_full_epoch036.pth results/run1_s4_baseline_ceiling/sgd_full_s{seed}_f{fold}/sgd_full_epoch037.pth results/run1_s4_baseline_ceiling/sgd_full_s{seed}_f{fold}/sgd_full_epoch038.pth results/run1_s4_baseline_ceiling/sgd_full_s{seed}_f{fold}/sgd_full_epoch039.pth results/run1_s4_baseline_ceiling/sgd_full_s{seed}_f{fold}/sgd_full_epoch040.pth
python src/run2_s5_cdan/eval_adabn.py --seed {seed} --fold {fold} --passes 1 --ckpt results/run2_s5_cdan/cdan_full_s{seed}_f{fold}/cdan_full_epoch036.pth results/run2_s5_cdan/cdan_full_s{seed}_f{fold}/cdan_full_epoch037.pth results/run2_s5_cdan/cdan_full_s{seed}_f{fold}/cdan_full_epoch038.pth results/run2_s5_cdan/cdan_full_s{seed}_f{fold}/cdan_full_epoch039.pth results/run2_s5_cdan/cdan_full_s{seed}_f{fold}/cdan_full_epoch040.pth
python src/run3_s5_iwcdan/eval_adabn.py --seed {seed} --fold {fold} --passes 1 --ckpt results/run3_s5_iwcdan/iwcdan_full_s{seed}_f{fold}/iwcdan_full_epoch036.pth results/run3_s5_iwcdan/iwcdan_full_s{seed}_f{fold}/iwcdan_full_epoch037.pth results/run3_s5_iwcdan/iwcdan_full_s{seed}_f{fold}/iwcdan_full_epoch038.pth results/run3_s5_iwcdan/iwcdan_full_s{seed}_f{fold}/iwcdan_full_epoch039.pth results/run3_s5_iwcdan/iwcdan_full_s{seed}_f{fold}/iwcdan_full_epoch040.pth
```

The DANN, MCC, and MDD trainers save the same fixed last five checkpoints. Their
paper-level scoring is handled by the separate evaluation package rather than
these five host-specific thin entries.

Exact scoring and cross-seed statistical aggregation are handled by the
evaluation package prepared separately from this training/configuration group.
