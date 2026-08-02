# Simulation-source validity assessment

This package documents why arm2 was selected as the semi-synthetic source
domain for the downstream domain-adaptation study. It does not define a single
"realism score." It combines two checks that guard against different failure
modes:

1. A low-level format probe asks whether synthetic and real images remain easy
   to separate using only grayscale histogram and thumbnail information.
2. A fixed sim-to-real task-transfer experiment asks whether reducing that
   separability also improves classification transfer to the target domain.

The first check can be gamed by blur or information destruction. The second
check prevents a low AUC from being treated as sufficient evidence of useful
simulation quality.

## Four frozen source arms

Every arm contains 90 airplane and 400 ship images.

| Arm | Change relative to the previous arm | Role |
|---|---|---|
| arm0 | Native baseline semi-synthetic source | Most-original reference |
| arm1 | Low-level intensity/contrast treatment and heuristic shadow | Historical low-level gate stage |
| arm2 | arm1 plus revised ship morphology and interior treatment | Selected source used downstream |
| arm3 | arm2 plus internal ship structure | More complex alternative |

The 90 airplane files are byte-identical in arm1, arm2, and arm3. Therefore
only the arm0-to-arm1 step changes both classes; the later steps change ships
only.

## Check 1: low-level format probe

The probe is run separately for airplane and ship. It never pools the two
object classes.

- Feature vector: 32-bin normalized grayscale histogram plus an 8x8 grayscale
  thumbnail, for 96 dimensions in total.
- Domain classification: logistic regression with C=1.0 and the liblinear
  solver.
- Balancing: synthetic and real samples are downsampled without replacement
  to the smaller domain count within each class.
- Repetition: 20 repeated balanced resamples.
- Validation inside each resample: five-fold stratified cross-validation with
  seed 42.
- Output: 100 correlated fold-level AUC values per class. The stored standard
  deviation is descriptive across those 100 values; it is not a standard
  error or confidence interval.

This protocol must not be described as one conventional "100-fold
cross-validation." Its exact meaning is 20 balanced resamples times five
folds.

### Historical acceptance gate

The ship format-probe threshold of AUC <= 0.85 was a prospective internal gate
for the low-level brightness/contrast stage represented by arm1. It was not
re-applied as a new pass/fail gate to arm2 or arm3; those later shape and
structure variants were interpreted directionally. Passing this threshold
means only that this 96-dimensional probe is no longer highly discriminative.
It does not establish physical realism.

## Check 2: fixed-target task transfer

For each arm, the runner performs 12 complete source-only training runs using
matched seeds 0 through 11.

- Model: ImageNet-pretrained ResNet-18 with a two-class head.
- Optimizer: Adam, learning rate 1e-4, weight decay 1e-4.
- Loss: class-balanced weighted cross-entropy computed from the current source
  training set.
- Training: 40 fixed epochs, batch size 32, no target validation and no target
  label-based early stopping.
- Prediction: mean softmax probability from the final five epoch checkpoints,
  followed by argmax.
- Target evaluation: the same fixed 553-image KLSG-II set for every arm and
  seed, with 66 airplane and 487 ship images.

The target labels are used for the frozen final E2 scoring. They are not used
inside a training run for validation, checkpoint selection, or parameter
tuning.

The primary uncertainty calculation uses one difference per matched seed:

    d_seed = metric(arm_y, seed) - metric(arm_x, seed)

The report gives a two-sided 95% paired-t interval over the 12 differences. A
10,000-resample paired bootstrap over the same 12 differences is shown only as
a cross-check. There is no multiplicity correction. Primary and secondary
metrics must therefore remain visibly separated, and a confidence interval
that includes zero is called inconclusive rather than "no effect" or
"underpowered."

## Quantitative evidence

The full-source trajectory is:

| Arm | Ship format AUC | Airplane format AUC | Macro-F1 mean | Ship-F1 mean | Airplane recall mean |
|---|---:|---:|---:|---:|---:|
| arm0 | 0.9432 | 0.8164 | 0.3546 | 0.4542 | 0.8434 |
| arm1 | 0.8228 | 0.8229 | 0.5134 | 0.7118 | 0.7311 |
| arm2 | 0.7247 | 0.8229 | 0.5484 | 0.7837 | 0.6010 |
| arm3 | 0.7295 | 0.8229 | 0.5772 | 0.7972 | 0.6856 |

The direct arm0-to-arm2 evidence for the selected source is:

| Metric | Mean matched difference | 95% paired-t interval |
|---|---:|---:|
| Macro-F1 | +0.1938 | [0.1233, 0.2642] |
| Ship F1 | +0.3294 | [0.2077, 0.4512] |
| Ship recall | +0.3614 | [0.2524, 0.4704] |
| Airplane F1 | +0.0581 | [0.0333, 0.0829] |
| Airplane recall | -0.2424 | [-0.3172, -0.1676] |
| Balanced accuracy | +0.0595 | [0.0213, 0.0977] |

The separate real-to-real five-fold supervised reference has macro-F1
0.889547, rounded to 0.890 in the E2 report. The arm2 gap to that reference is
0.3416, down from 0.5354 for arm0. The airplane-recall decrease is a material
class trade-off and must accompany the aggregate improvement.

### Arm2 versus arm3

Arm2 was not selected because it dominated arm3 on task performance. Arm3 has
the highest absolute macro-F1 in this fixed target experiment. For arm3 minus
arm2, the prospectively designated ship metrics are inconclusive:

| Comparison | Mean matched difference | 95% paired-t interval |
|---|---:|---:|
| Full-source ship F1 | +0.0136 | [-0.0404, 0.0675] |
| Full-source ship recall | +0.0111 | [-0.0709, 0.0931] |
| Changed-263 ship F1 | +0.0430 | [-0.0068, 0.0929] |
| Changed-263 ship recall | +0.0501 | [-0.0116, 0.1119] |

The secondary full-source balanced-accuracy difference favors arm3
(+0.0479, interval [0.0169, 0.0789]). The changed-image secondary macro-F1
interval also narrowly excludes zero. These results are retained as
counterevidence rather than hidden.

Arm2 was selected because it had the lowest ship format-probe AUC, the primary
ship-specific arm2-to-arm3 increments were not established by the fixed rule,
and arm2 omitted an additional structure module. This is a parsimonious,
target-aware choice under this protocol, not a claim that arm2 is universally
more realistic or transfers best.

## Structure-change subset

The historical internal label "268 subset" was stale. A fresh byte comparison
finds 263 ship files that differ between arm2 and arm3 and 137 no-op ship
files. The subset experiment trains each arm on all 90 airplanes plus those
263 changed ships, for 353 training images, and still evaluates on the full
553-image target set. It is a source-training subset, not a target-test
subset.

## Evidence layout

The tracked evidence directory is evidence/simulation_validity:

- per_run: 48 full-source JSON records, one per arm and seed.
- per_run_structure_subset: 24 JSON records for arm2 and arm3.
- format_probe_arm0.json through format_probe_arm3.json.
- structure_subset_ship_fnames.json.
- summary_arm0.md through summary_arm3.md.
- paired_delta_report.md and paired_delta_report.json.
- ceiling_reference.csv.
- SHA256SUMS.txt covering every other file in this evidence directory.

The 72 internal text logs were inspected before packaging. They contained no
additional metric, convergence, or error information beyond the structured
JSON records and did contain machine-local paths and timestamps, so they are
not included in the public evidence set.

Every per-run JSON retains the training losses, confusion matrix, metrics,
software versions, target input-byte fingerprint, and NaN diagnostic. In all
72 frozen cells, final loss was below initial loss and no NaN was observed.
The legacy plateau heuristic was false for all 72 cells; it was too strict for
these low-loss trajectories and was never used to filter a run.

## Recomputing the statistics

The included evidence is sufficient to reproduce all paired differences,
intervals, bootstrap cross-checks, arm summaries, and ceiling-gap values
without the image datasets:

    python src/simulation_validity/analyze_paired_delta.py

Strict mode refuses to analyze unless it finds exactly 48 full cells, exactly
24 arm2/arm3 structure-subset cells, the exact 263/137 structure manifest, all
four complete format-probe files, and one shared non-empty target input-byte
fingerprint across all 72 cells. It also checks internal arm/seed identities,
training and target counts, finite losses and metrics, and exact agreement
between every confusion matrix and its stored metrics. It requires one common
recorded PyTorch/CUDA/Python environment for the historical cells and verifies
the full normalized environment signature when new-run fields are present. For the bundled
evidence, strict mode verifies SHA256SUMS.txt before analysis. It writes
paired_delta_report_recomputed.md and the corresponding JSON at the repository
root, outside the immutable evidence directory.

## Re-running the full experiment

Full reruns additionally require all four source-arm datasets, KLSG-II under
its upstream terms, and access to the pretrained ResNet-18 weights. No
third-party KLSG-II images are included here.

Replace every PATH_TO_* placeholder near the top of
e2_ablation_runner.py. Each arm and KLSG root must contain airplane and ship
subdirectories. The canonical evaluation root must contain grayscale 224x224
PNG files; the raw KLSG root is used by the format probe.
The repository does not include the raw-to-canonical KLSG preparation script,
so the raw target tree alone is not sufficient for a full E2 rerun.

To run only the low-level probe:

    python src/simulation_validity/e2_ablation_runner.py --format_probe_only --arms 0,1,2,3

To run the complete 48-cell E2 experiment and the 24 structure-subset cells:

    python src/simulation_validity/e2_ablation_runner.py --arms 0,1,2,3 --structure_subset

Generated files are written only below the configured output directory.
In normal training mode, --arms selects which full-data cells are trained; the
runner still validates and fingerprints the complete four-arm frozen input
contract. --structure_subset is a paired arm2/arm3 comparison and therefore
requires both arms 2 and 3 in --arms. Partial seed runs do not rewrite an arm
summary.

Before training, the runner fingerprints every frozen source arm with SHA-256
and the target input-byte set with the legacy MD5 manifest used by the
experiment. Existing per-run JSON records are reused only when the arm, seed,
epoch count, training subset, source fingerprint, target fingerprint,
structure-subset metadata, normalized protocol SHA-256, and runner-source
SHA-256 match. New records also bind Python, PyTorch, torchvision, NumPy,
OpenCV, CUDA, cuDNN, platform, and device metadata through a normalized
software-environment SHA-256. A mismatch fails loudly and requires an explicit
--force rerun.
The runner also verifies that arm1/arm2/arm3 airplane filenames and bytes are
identical and that arm2/arm3 ship filename sets match before identifying the
263 changed ships. These input contracts use explicit runtime errors and remain
active when Python is launched with optimization (`python -O`). Format probes
are always recomputed, so a stale probe JSON
cannot be reused silently; newly generated probe JSON records SHA-256
fingerprints for both its source arm and raw KLSG-II inputs, plus its own
Python/NumPy/OpenCV/scikit-learn environment fingerprint. JSON and summary
outputs are atomically replaced only after a complete new file has been written.

Configured absolute input paths are used to read local data but are not
persisted in the normal JSON or summary outputs; those files store logical
dataset identifiers. The structure-subset manifest also stores a SHA-256 over
its sorted filename set.

## Interpretation limits

- KLSG-II images informed qualitative physical and morphological review, then
  the same target dataset was used for the format probe, E2 scoring, and arm2
  selection. This is target-aware source selection, not an untouched external
  validation set.
- Reusing arm2 and KLSG-II in the downstream benchmark supports conditional
  comparisons among methods that share the frozen source and target. It does
  not establish generalization to an unseen real domain.
- The paired intervals quantify training-seed variation conditional on one
  fixed target image set. They do not include target-dataset sampling
  uncertainty.
- The format-probe cross-validation is image-level and has no wreck-, site-,
  or acquisition-group split.
- The frozen format-probe JSON files retain the mean and descriptive standard
  deviation of the 100 correlated fold-level AUC values, not the 100 values
  themselves. Recomputing those values requires the source and KLSG-II images.
- The frozen 72 per-run records predate the public runner's source-arm
  SHA-256, protocol-signature, runner-source-signature, and full
  software-environment-signature fields. They bind all cells to one target
  image-byte fingerprint and retain logical source identifiers, counts, and
  common PyTorch/CUDA/Python versions, but do not cryptographically bind the
  historical cells to source-arm bytes, a public code snapshot, or the complete
  dependency environment. New reruns record those additional fingerprints.
- An AUC near 0.5 would indicate chance performance only for this 96-feature
  probe. It would not prove physical equivalence.
- The protocol was prospectively frozen in internal project records before
  the 48 formal cells were run, but it was not registered in an external,
  immutable registry.
- No claim of universal physical realism or cross-dataset validity follows
  from this evidence.
