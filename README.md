# Side-scan sonar sim-to-real domain adaptation

Code, configuration records, versioned data metadata, and structured evidence
accompanying the manuscript:

> Luyu Ren and Benjun Ma, "A cost-aware, minority-class evaluation of
> adversarial domain adaptation for side-scan sonar sim-to-real
> classification."

The repository covers the final semi-synthetic source-generation pipeline,
data loading and split metadata, source-only and domain-adaptation training,
fixed last-K evaluation, the simulation-source validity assessment, and the
complete ASDA comparator with the records of its archived runs. It is a
research artifact and reference implementation, not a zero-configuration or
one-command reproduction package.

## Repository contents

| Path | Contents |
|---|---|
| `src/simulation/` | Final arm2 semi-synthetic source-generation snapshot |
| `src/datapreprocess/` | Five-fold split construction |
| `src/run1_s4_baseline_ceiling/` | Source-only baseline, supervised ceiling, and AdaBN evaluation |
| `src/run2_s5_cdan/` | CDAN and entropy-conditioned CDAN |
| `src/run3_s5_iwcdan/` | IW-CDAN and AdaBN evaluation |
| `src/run4_s5_iwcdan_frozen/` | Frozen-BN IW-CDAN diagnostic configurations |
| `src/run5_s5_iwcdan_dsbn/` | DSBN IW-CDAN |
| `src/extra/` | DANN, MCC, and MDD supplementary candidates |
| `src/extra/asda/` | Complete ASDA comparator, implemented independently from the published method |
| `src/evaluation/` | Frozen scoring, collapse diagnostics, paired statistics, and multiplicity correction |
| `src/simulation_validity/` | Format probe, fixed-target E2, and paired simulation-source analysis |
| `configs/` | Frozen experiment matrix and simulation-validity contract |
| `data/` | Dataset metadata, integrity records, and split manifests; no third-party image bytes |
| `evidence/simulation_validity/` | Structured evidence supporting the selected simulation source |
| `evidence/asda/` | Structured records of the 25 archived ASDA runs |
| `tests/` | Public regression and evidence-integrity tests |

## Environment

The main training and E2 experiments were run with Python 3.11.15, PyTorch
2.9.1+cu128, torchvision 0.24.1+cu128, CUDA 12.8, cuDNN 91002, and NumPy
2.2.6. `requirements.txt` records the verified Python package versions needed
by the published code. The CUDA build is not portable to every operating
system or GPU, so select a compatible PyTorch build for the local platform if
the original CUDA 12.8 stack is unavailable.

The pinned environment targets CPython 3.11. The original experiments used
Conda; the following commands create an interpreter independently of the
system-wide `python` command:

```text
conda create --name sss-sim2real python=3.11.15
conda activate sss-sim2real
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a Python 3.11 interpreter is already installed, its explicit executable
path may instead be used with `-m venv .venv`; verify that the activated
environment reports Python 3.11.x before installing the pinned requirements.

For a CUDA-specific installation, install the matching PyTorch and torchvision
wheels using the [official PyTorch selector](https://pytorch.org/get-started/locally/),
then install the remaining requirements. The optional `skvideo` and `pyiqa`
backends in `src/evaluation/image_diagnostics.py` require `scikit-video` and
`pyiqa`, respectively; neither backend is needed for the paper's core scoring
or paired comparisons.

## Data

### Final semi-synthetic source

The manuscript used a 490-image semi-synthetic source domain with 90 airplane
and 400 ship grayscale PNG images. This code repository does not distribute
those image assets. [`data/manifest.csv`](data/manifest.csv) records the
versioned filenames, classes, dimensions, byte sizes, and SHA-256 digests for
the source used in the study; it is metadata only and does not grant any right
to obtain, reuse, or redistribute the underlying images. See
[`data/README.md`](data/README.md) for the data-access boundary.

To use the training scripts, place source images that you are entitled to use
in this repository layout:

```text
data/
  raw/
    ai4simulate/
      airplane/
      ship/
```

### Real target domain

The study used SeabedObjects-KLSG-II obtained from the authors' public
[provenance repository](https://github.com/HHUCzCz/-SeabedObjects-KLSG--II)
before its contents changed. At the time of this release, that repository no
longer contains the complete 66-airplane/487-ship dataset and cannot be used to
reconstruct the paper's target set. The repository also does not state a
redistribution licence, so no KLSG-II image bytes are included here. A user who
lawfully obtains the complete dataset from its authors or another authorized
source should place the two class directories as follows:

```text
data/
  raw/
    SeabedObjects/
      airplane/
      ship/
```

### Simulation inputs

The source generator uses modified material or statistics from
[AI4Shipwrecks](https://doi.org/10.7302/dmf4-x492), which is licensed under
CC BY 4.0. Original AI4Shipwrecks files are not redistributed. The
target-highlight texture and seabed brightness statistics were fitted to
AI4Shipwrecks; KLSG-II did not supply these statistics, and none of the
third-party silhouette inputs was derived from KLSG-II images. This concerns
the provenance of the synthesis inputs; KLSG-II was used to assess and develop
the source, as described under Core evaluation below. The archived
reference pipeline also retains preparation stages for locally supplied
Marine-PULSE/URM inputs. A full regeneration additionally requires local
airplane/ship silhouette inputs, the curated `relook_strict.csv` frame
manifest, and the baseline intermediate products identified by the staged
scripts. None of those local inputs is tracked here. Obtain upstream inputs
separately under their own terms and follow
[`src/simulation/README.md`](src/simulation/README.md). The published code is
therefore a provenance and method snapshot, not a self-contained rebuild of
the manuscript's 490-image source domain.

## Training

The trainers infer the repository root from their own file locations. They use
the `data/raw/ai4simulate` and `data/raw/SeabedObjects` layout above and do not
contain a fallback to an author-machine path.

To start one source-only paper run from the repository root:

```text
python src/run1_s4_baseline_ceiling/train_source_only_sgd.py --epochs 40 --seed 0 --fold 0
```

To start one CDAN run:

```text
python src/run2_s5_cdan/train_cdan_sgd.py --epochs 40 --seed 0 --fold 0
```

Every paper configuration, the seed/fold ranges, supervised-ceiling
authorization command, last-K evaluation commands, and AdaBN commands are in
[`configs/README.md`](configs/README.md). The machine-readable descriptive
contract is [`configs/experiment_matrix.yaml`](configs/experiment_matrix.yaml);
it records the frozen settings but is not loaded automatically by the trainers.

## Core evaluation

The primary prediction rule averages the softmax probabilities from exactly
five fixed checkpoints and then applies argmax. Final target labels are read
only after training and model selection have been frozen.

`src/evaluate.py` is a low-level scoring and aggregation interface. Its `score`
subcommand expects a caller-prepared `[5,N,2]` NumPy array containing softmax
probabilities in fixed checkpoint and target-image order, plus a matching `[N]`
integer label array with class order `airplane=0, ship=1`. The published
training and last-K wrappers do not export these arrays, so users must prepare
them in their own inference/export step. To score one arm/seed/fold once those
inputs exist:

```text
python src/evaluate.py score --checkpoint-probabilities PATH_TO_PROBABILITIES.npy --labels PATH_TO_LABELS.npy --authorize-target-labels --authorization-reason "post-freeze final scoring" --arm source_only_adabn --seed 0 --fold 0 --output PATH_TO_SCORE.json
```

The `score` command is the strict paper-scoring contract and rejects anything
other than exactly five checkpoints. The host-specific `eval_lastk.py` and
`eval_adabn.py` wrappers are label-blind health/post-processing helpers and can
operate on an incomplete checkpoint set; such output is not valid formal
paper scoring.

The `collapse` subcommand additionally requires an epoch-count JSON record, and
`compare` requires the complete set of collapse-enriched arm/seed/fold records.
These subcommands implement the three-route collapse decision, seed/fold
aggregation, paired intervals, and Holm and Benjamini-Hochberg corrections;
they are not a batch exporter for the training output directories. Their
complete argument contract is available through:

```text
python src/evaluate.py --help
python src/evaluate.py score --help
python src/evaluate.py collapse --help
python src/evaluate.py compare --help
```

The split metadata and exact target-label policy are documented in
[`data/split/README.md`](data/split/README.md). Target labels are forbidden in
UDA losses, target-weight estimation, model selection, and early stopping.
Within the main benchmark, they are used for supervised-ceiling fold
construction, supervised-ceiling training and held-out scoring, and explicit
post-freeze final scoring. Separately, the simulation-source validity study
uses KLSG-II labels to group the real images for the class-wise format probe,
and for frozen E2 scoring and the resulting target-aware arm2 selection;
KLSG-II images also informed visual morphology decisions during source
development. That limitation is documented below.

## Source generation and simulation-validity evidence

The final source-generation snapshot is intentionally a staged reference
implementation. Before running it, replace the explicit `PATH_TO_*`
placeholders in the simulation scripts with valid local paths. The exact stage
order and final 90-airplane/400-ship merge are given in
[`src/simulation/README.md`](src/simulation/README.md).

The simulation-validity runner uses two distinct KLSG-II inputs: the raw class
directories for the format probe and a separately prepared canonical tree of
grayscale 224x224 PNG files for fixed-target E2. The repository does not
include the raw-to-canonical preparation script, so the `data/raw/SeabedObjects`
layout alone is not sufficient for a full E2 rerun. However, all paired
statistics and summary values can be recomputed from the tracked structured
evidence without any image datasets:

```text
python src/simulation_validity/analyze_paired_delta.py
```

See [`src/simulation_validity/README.md`](src/simulation_validity/README.md) for
the format-probe protocol, 12-seed fixed-target E2 design, quantitative results,
arm2 selection rationale, counterevidence, and interpretation limits.

## ASDA comparator

[`src/extra/asda/`](src/extra/asda/) contains the complete ASDA comparator
reported in the manuscript, implemented independently from the equations and
Algorithm 1 of Gou and Cui (2026). Its
[README](src/extra/asda/README.md) gives the implementation conventions of
manuscript Table E.1, the target-label boundary, the commands for the 25
seed/fold runs, and the relation between the public and the executed code.
[`evidence/asda/`](evidence/asda/) holds the structured records of the 25
archived runs. The manuscript's values for the complete comparator (Table 6,
Tables E.2 and E.3, and its cost records) can be recomputed from them without
images or a GPU:

```text
python -m unittest tests.test_asda -v
```

ASDA checkpoints and prediction arrays, and the code of the post-hoc ASDA
diagnostics in Appendices E.3 and E.4 of the manuscript, are not included.

## Tests

Run the public test suite from the repository root:

```text
python -m unittest discover -s tests -v
```

The tests use synthetic arrays and the tracked evidence files; they need no
image data or GPU.

The repository does not include KLSG-II images, original AI4Shipwrecks files,
the four simulation-source arm trees, training checkpoints, or the main paper
experiment outputs (apart from the structured ASDA run records and the
seed-level values of two reference arms in `evidence/asda/`). GPU-heavy
training and format-probe reruns therefore
require separately obtained inputs and suitable hardware.

## Versions

- v1.1.0 adds the complete ASDA comparator (`src/extra/asda/`), the records of
  its 25 archived runs (`evidence/asda/`) and `tests/test_asda.py`, and updates
  the documentation accordingly.
- v1.0.1 keeps the frozen evidence files in LF line endings on checkout.
- v1.0.0 is the initial public code release.

## Licence and citation

Source code and accompanying documentation are released under the
[MIT License](LICENSE). This repository grants no licence for non-included
image data or other external inputs. Portions adapted from upstream
MIT- or BSD-licensed implementations retain their notices in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Citation metadata for this
repository is provided in [`CITATION.cff`](CITATION.cff).
