# Examples

Worked examples of the three pieces of `lgctk`: a general one for your own CSV file and two on specific datasets.
They are **not part of the package API**: they show one recommended way of building segments, partitions and the
LGC scope. The decisions marked below belong to the scientist. No data are included: Team Metrics is a private
dataset and DEAP is not redistributed (request it from its authors).

Requirements: the package plus pandas (`pip install lgc-toolkit pandas`) and a clone of this repository, which
contains the examples. Run from the repository root, e.g.
`python examples/csv_quickstart.py --data eeg_dataset.csv --channels Fp1 Fp2 --fs 128`.

| Example | Pipeline | Data access | What you decide |
|---|---|---|---|
| `csv_quickstart.py` | LGC symmetric, then RCT + MDM or TSA-LR, leave-one-subject-out | target without labels, or labelled calibration with a held-out test | channels, real fs, windows, K, calibration/test design |
| `team_metrics_transductive.py` | [LGC symmetric / random / none] + RCT + MDM, leave-one-subject-out | target covariances without labels; every target window predicted | band, K, LGC kind and seed |
| `team_metrics_calibrated_tsa.py` | [LGC] + TSA-LR, leave-one-subject-out | target blocks 1–6 calibrate (labels used), 7–8 test (no labels) | which recorded blocks calibrate |
| `deap_stratified_split.py` | [LGC] + TSA-LR, leave-one-subject-out | per-trial stratified calibration/test split of each target | the split rule and its seed |

These are usage examples, not exact reproductions of the reported figures ([README](../README.md#results),
[VALIDATION.md](../VALIDATION.md)).

## Input formats
- **Your own CSV** (`csv_quickstart.py`, start here): one row per sample with the columns `subject`, `segment`,
  `label` and the EEG channels; walkthrough in the [README quick start](../README.md#quick-start). The script checks
  the input (columns, missing identifiers, numeric and finite channels, contiguous segments and subjects, one label
  per segment, band below Nyquist); it does not check the physical units or the real sampling rate of the file.
  `--make-synthetic eeg_dataset.csv` writes a synthetic file in this format (not real EEG; its accuracies mean
  nothing).
- **Team Metrics** (`_team_metrics_io.py`): one CSV, one row per sample at 128 Hz, columns = 32 EEG channels (`Fp1`
  … `Cz`), `time`, `block` (the RECORDED block identifier), `class` (condition of the block) and `subject`; the rows of
  each block in temporal order. Specific to this dataset; for other data use `csv_quickstart.py`.
- **DEAP**: request access from its authors (https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), download the
  **preprocessed MATLAB release**, keep the folder `data_preprocessed_matlab` (`s01.mat` … `s32.mat`: `data` 40
  trials × 40 channels × 8064 samples at 128 Hz, the first 32 channels EEG; `labels` 40 × 4 ratings) on your machine
  and pass `--data /path/to/data_preprocessed_matlab`. The example reads the files with `scipy.io.loadmat`; it does
  not read the Python (pickle) release.

No real recording, label or file of either dataset is part of this repository; the tests use synthetic data only.

## Preprocessing and parameters
The examples use the optional utilities of `lgctk.preprocessing` -- zero-phase Butterworth band-pass of order 4,
sliding windows and Ledoit-Wolf covariances -- applied **one recorded segment at a time**
([docs/METHOD.md](../docs/METHOD.md#preprocessing-optional-utilities)). You may replace them with your own
preprocessing and pass covariances to the core functions. Parameters you can change on the command line:

### CSV quick start
- `--channels` and `--fs` are required for your own data.
- `--band` takes two frequency bounds (default: 8 13 Hz).
- `--window-sec` and `--step-sec` default to 4 and 2 seconds.
- `--half-window` sets symmetric LGC K (default: 10).
- Choose `--route unlabelled`, `--route acquisition-order` or `--route stratified`. The default `--route all` runs
  all three and therefore requires data suitable for both calibration designs; `--calibration-segments` defaults
  to 6.

### Dataset-specific examples

| Option | Default (validated configuration) | Meaning |
|---|---|---|
| `--band` (Team Metrics) | alpha 8–13 Hz | band-pass; DEAP uses 15–36 Hz |
| `--window-sec`, `--step-sec` | 4 s, 2 s (50 % overlap) | converted to samples as `seconds × fs` (fs = 128 Hz) |
| `--half-window` | 10 | LGC radius K in **windows**; see the note on K below |
| `--lgc`, `--seed` | symmetric, 42 | LGC kind; seed of random LGC |

K is counted in windows: with a 2 s step, symmetric K = 10 spans ±20 s; for random LGC, K only sets how many
neighbours are drawn ([docs/METHOD.md](../docs/METHOD.md#symmetric-neighbourhood-lgc_symmetric)). Other values do
not inherit the reported figures.

## Decisions that belong to the scientist
1. **Segments.** `segment_ids` come from acquisition metadata: the recorded block (Team Metrics) or the trial index of
   the file (DEAP). Never compute them from labels ([docs/PROTOCOLS.md](../docs/PROTOCOLS.md#segments-and-metadata)).
2. **Calibration/test partition** (calibrated examples). Assigned per segment, built **once, before evaluation** and
   then frozen. In Team Metrics it comes from metadata (block numbers). In DEAP the example stratifies by class: it
   uses the per-trial labels of the target, test trials included, as a declared benchmark construction, outside the
   predictive function, which never receives test labels. The split seed (`--split-seed`) is separate from the LGC
   seed (`--seed`) and from the TSA fit seed.
3. **LGC scope.** Sources: LGC over the whole subject. Target: LGC applied separately to its calibration part and to
   its test part, passed as `target_covs`, so that no neighbourhood crosses from one trial or block to another or
   mixes calibration with test. LGC of the test part uses test covariances of the same segment: offline
   preprocessing, not a fit.

See [docs/PROTOCOLS.md](../docs/PROTOCOLS.md) for what each stage may access.
