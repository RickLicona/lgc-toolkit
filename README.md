# lgc-toolkit

**Local Geometric Consistency for Cross-Subject EEG Transfer Learning**

A Python implementation of Local Geometric Consistency (LGC) for cross-subject EEG classification in passive
brain–computer interfaces. LGC aggregates neighbouring covariance matrices through Riemannian averaging within
recording segments.

The toolkit provides symmetric and random neighbourhoods and supports integration with the Riemannian Centering
Transformation followed by a minimum-distance-to-mean classifier (RCT + MDM) and with calibrated Tangent Space
Alignment followed by logistic regression (TSA-LR).

<!-- Zenodo DOI badge goes here after the deposit:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.<concept id>.svg)](https://doi.org/10.5281/zenodo.<concept id>) -->
Version `0.1.0.dev0` (development release). Distribution name `lgc-toolkit`, import name `lgctk`.

---

## Pipeline

Notation, used consistently throughout the diagram:
- C: number of EEG channels.
- T_total: number of samples per recording segment.
- T: number of samples per window.
- N: total number of windows across subjects and recording segments.
- [f_lo, f_hi]: frequency band in Hz (e.g. 8–13 Hz or 15–36 Hz).

```
                      Preprocessed EEG
            Per recording segment: (C, T_total)
                             │
                             ▼
       ┌──────────────────────────────────────────┐
       │           Band-pass filtering            │
       │           Butterworth order 4            │
       │       Frequency band [f_lo, f_hi]        │
       │          Example: alpha 8–13 Hz          │
       └──────────────────────────────────────────┘
                             │
                             ▼
       ┌──────────────────────────────────────────┐
       │             Sliding windows              │
       │       Within each recorded segment       │
       │        Example: 4 s · 50% overlap        │
       └──────────────────────────────────────────┘
                             │  (N, C, T)
                             ▼
       ┌──────────────────────────────────────────┐
       │          Covariance estimation           │
       │               Ledoit–Wolf                │
       └──────────────────────────────────────────┘
                             │  (N, C, C)
                             ▼
       ╔══════════════════════════════════════════╗
       ║    LGC — Local Geometric Consistency     ║
       ║           Riemannian averaging           ║
       ║    Symmetric or random neighbourhoods    ║
       ║         Within recorded segments         ║
       ╚══════════════════════════════════════════╝
                             │  (N, C, C)
              ┌──────────────┴───────────────┐
              ▼                              ▼
     UNSUPERVISED DOMAIN             SUPERVISED DOMAIN
          ADAPTATION                    ADAPTATION
      Unlabelled target         Labelled target calibration
              │                              │
              ▼                              ▼
 ┌────────────────────────┐   ┌────────────────────────────┐
 │          RCT           │   │  Tangent Space Alignment   │
 │  Riemannian centering  │   │ Tangent-space projection,  │
 │    of each subject     │   │   centering, scaling and   │
 │                        │   │    supervised rotation     │
 └────────────────────────┘   └────────────────────────────┘
              │                              │
              ▼                              ▼
 ┌────────────────────────┐   ┌────────────────────────────┐
 │          MDM           │   │    Logistic regression     │
 │    Labelled sources    │   │      Labelled sources      │
 │          only          │   │    + target calibration    │
 └────────────────────────┘   └────────────────────────────┘
              │                              │
              ▼                              ▼
     Classify all target             Classify held-out
           windows                     test windows
              │                              │
              └──────────────┬───────────────┘
                             ▼
       ┌──────────────────────────────────────────┐
       │     Leave-one-subject-out evaluation     │
       │           Accuracy per subject           │
       └──────────────────────────────────────────┘
                             │
                             ▼
       ┌──────────────────────────────────────────┐
       │      Mean accuracy across subjects       │
       └──────────────────────────────────────────┘
```

## Methods

| Method | Components | Target data used for fitting | Evaluator |
|---|---|---|---|
| RCT + MDM | `RCTMDM` | covariances without labels (unsupervised re-centring) | `loso_unlabelled_target` |
| LGC + RCT + MDM | `lgc_symmetric` or `lgc_random`, then `RCTMDM` | covariances without labels (unsupervised re-centring) | `loso_unlabelled_target` |
| TSA-LR | `TSALR` | calibration windows with labels; test windows only predicted | `loso_calibrated` |
| LGC + TSA-LR | `lgc_symmetric` or `lgc_random`, then `TSALR` | calibration windows with labels; test windows only predicted | `loso_calibrated` |

- **LGC.** For window $i$ with covariance $C_i$, $\bar C_i$ is the Riemannian (affine-invariant) mean of the
  covariances in its neighbourhood. The symmetric neighbourhood holds up to $K$ windows on each side of $i$, clipped at
  the segment edges. The random neighbourhood has the same size and contains $i$ plus windows drawn without
  replacement from the same segment (the seed is mandatory). For symmetric LGC, K is a radius in windows: with a 2 s
  step, K = 10 reaches up to ±20 s between window starts, clipped at the segment boundaries. For random LGC, K sets
  neighbourhood size; neighbours may come from anywhere in the same segment. Neighbourhoods include future windows,
  so LGC is an **offline** method.
- **RCT + MDM** [1, 2]: each subject is re-centred to the identity at its own Riemannian mean; MDM (log-Euclidean
  class means, Riemannian distance) is fitted on the source subjects.
- **TSA-LR** [3]: tangent-space alignment with per-domain centring and scaling and a supervised rotation of the
  sources towards the target calibration, followed by logistic regression. The original TSA work uses an SVM.

Details and formulas: [docs/METHOD.md](docs/METHOD.md).

## Evaluation protocols

Cross-subject transfer within one dataset, leave-one-subject-out.

- **Target without labels** (`loso_unlabelled_target`): the target's covariances, without labels, enter the
  unsupervised re-centring; every target window is predicted.
- **Labelled target calibration** (`loso_calibrated`): a partition assigns each recorded segment of the target to
  `"calibration"` or `"test"`. Calibration windows (with labels) and source windows enter every fitted stage; test
  windows are only predicted.

In both, the predictive function never receives the evaluation labels: only the evaluator holds them. Segment
identifiers must come from acquisition metadata (recorded block or trial), never from labels. The partition is the
scientist's benchmark design and is validated before any fit. See [docs/PROTOCOLS.md](docs/PROTOCOLS.md).

## Installation

```bash
pip install git+https://github.com/RickLicona/lgc-toolkit
```

<!-- PyPI: after publishing, make this the first option:
```bash
pip install lgc-toolkit
```
-->

To run the tests and the worked examples, clone the repository:

```bash
git clone https://github.com/RickLicona/lgc-toolkit
cd lgc-toolkit
pip install ".[test]" pandas
python -m pytest
```

### Dependencies

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| numpy | ≥ 1.26.4 |
| scipy | ≥ 1.13.1 |
| scikit-learn | ≥ 1.4.2 |
| pyriemann | 0.9 |
| threadpoolctl | ≥ 3.5.0 |
| pandas (examples only) | ≥ 2.2.2 |

Tested with NumPy 1.26.4 and 2.0.2, scikit-learn 1.4.2 and 1.6.1. The reference results were produced with Python
3.9.16, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.4.2 and PyRiemann 0.9
([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)).

## Quick start

The walkthrough below goes from a CSV file to the two evaluation routes. The complete script, with input checks, is
[examples/csv_quickstart.py](examples/csv_quickstart.py). To try it without data of your own, write a synthetic
recording in the expected format (2 channels, 128 Hz, 3 subjects, 8 trials of 30 s each):

```bash
python examples/csv_quickstart.py --make-synthetic eeg_dataset.csv
python examples/csv_quickstart.py --data eeg_dataset.csv --channels Fp1 Fp2 --fs 128
```

### 1. The input file

One row per **sample** (not per window), in temporal order:

| subject | segment | label | Fp1 | Fp2 |
|---|---|---|---|---|
| S01 | trial_01 | 0 | 0.3232 | -0.1845 |
| S01 | trial_01 | 0 | 0.4150 | -0.0921 |
| S01 | trial_02 | 1 | -0.2710 | 0.5382 |
| S02 | trial_01 | 0 | 0.1984 | -0.3025 |

*Illustrative EEG amplitudes in µV; intermediate samples are omitted.*

- `subject`: participant identifier.
- `segment`: the recorded trial or block, unique within the subject (it may repeat across subjects). With several
  sessions, use identifiers that never merge different trials.
- `label`: the condition, here one integer per segment (0/1 with the same meaning for every subject).
- `Fp1`, `Fp2`: example channels; use any channels, in the same order and with the same meaning for every subject.
- Amplitudes in consistent units across subjects and channels (check your export; nothing is converted).
- One real, common sampling rate. Declaring `fs = 128` does not resample a 256 Hz file.

Segments come from acquisition metadata: never build them from label changes. The constant label per segment is a
condition of this example, not a definition of an LGC segment. If your file uses other column names, pandas can
rename and select them (the original names below are only an example; reading your original format, and recovering
missing event markers, depends on your dataset):

```python
import pandas as pd

data = pd.read_csv("original_recording.csv")
data = data.rename(columns={"participant": "subject", "trial": "segment", "condition": "label"})
channels = ["Fp1", "Fp2"]
data[["subject", "segment", "label"] + channels].to_csv("eeg_dataset.csv", index=False)
```

### 2. Windows, covariances and LGC

You declare the channels, the real sampling rate, the window length and the step; windows, covariances, positions
and window IDs are built per segment. `bandpass`, `sliding_windows`, `lw_covariances` and `lgc_symmetric` are package
functions; the loop is example code (`prepare` in the script, which adds input checks).

```python
import numpy as np
import pandas as pd
from lgctk import RCTMDM, TSALR, lgc_symmetric, loso_calibrated, loso_unlabelled_target
from lgctk.preprocessing import bandpass, lw_covariances, sliding_windows

channels = ["Fp1", "Fp2"]
fs = 128                 # the real sampling rate of the file
window_sec, step_sec = 4, 2
data = pd.read_csv("eeg_dataset.csv")

covariances, metadata = [], []
for (subject, segment), rows in data.groupby(["subject", "segment"], sort=False):
    eeg = rows[channels].to_numpy(dtype=float).T                       # (channels, samples)
    filtered = bandpass(eeg, lo=8, hi=13, fs=fs, axis=1, order=4)
    windows, starts = sliding_windows(filtered, window_len=round(window_sec * fs), step_len=round(step_sec * fs))
    covariances.append(lw_covariances(windows))
    label = rows["label"].iloc[0]
    metadata.extend((subject, segment, position, label) for position in range(len(windows)))

covs = np.concatenate(covariances)
meta = pd.DataFrame(metadata, columns=["subject", "segment", "position", "label"])
window_ids = np.arange(len(meta))
X = lgc_symmetric(covs, meta["subject"].to_numpy(), meta["segment"].to_numpy(), meta["position"].to_numpy(),
                  half_window=10)                                       # no labels involved
```

At 128 Hz this gives windows of 512 samples every 256 samples. These are example settings, not requirements; the
band must lie below the Nyquist frequency.

### 3. Route 1: target without labels

Leave-one-subject-out: each participant in turn is the target and the others are the sources. The target's
covariances, without labels, take part in the re-centring; all its windows are predicted and scored. The target is
therefore not held out from every fitted stage.

```python
result = loso_unlabelled_target(X, meta["subject"].to_numpy(), meta["label"].to_numpy(), RCTMDM(),
                                window_ids=window_ids)
print(f"Mean accuracy: {result.mean_accuracy:.1%}")
```

### 4. Route 2: labelled calibration and held-out test

```text
Fold 1: all participants except subject 1 are sources.
        Subject 1: labelled calibration -> fit; held-out test -> evaluation.
Fold 2: all participants except subject 2 are sources.
        Subject 2: labelled calibration -> fit; held-out test -> evaluation.
Repeat until every participant has been the target.
```

"Held-out" refers to the target's test part, not to the whole subject: its calibration part enters the fit. No
fitted stage receives the test covariances or the test labels. The test may still be processed offline with LGC
inside its own segment, so held-out does not mean causal. The partition assigns whole segments, is decided before
evaluating and then stays frozen. Two designs:

**A. By acquisition order.** The first six trials of each subject calibrate, the last two are the test. The order is
the order of the file (acquisition order), never a sort of the segment names. Calibration must contain every class.

```python
calibration = {
    (subject, segment)
    for subject, rows in meta.groupby("subject", sort=False)
    for segment in rows["segment"].unique()[:6]                         # first six trials, in file order
}
partition = np.array(["calibration" if key in calibration else "test"
                      for key in zip(meta["subject"], meta["segment"])])
```

**B. Stratified by trials.** About 25 % of each subject's trials go to test, stratified by class, with a fixed seed.
The split is made on the table of trials and then propagated to their windows: never split the windows themselves.

```python
from sklearn.model_selection import train_test_split

trials = meta[["subject", "segment", "label"]].drop_duplicates()
assignment = {}
for subject, rows in trials.groupby("subject"):
    calibration_ids, test_ids = train_test_split(rows["segment"].to_numpy(), test_size=0.25,
                                                 stratify=rows["label"].to_numpy(), random_state=20261012)
    assignment.update({(subject, s): "calibration" for s in calibration_ids})
    assignment.update({(subject, s): "test" for s in test_ids})
partition = np.array([assignment[key] for key in zip(meta["subject"], meta["segment"])])
```

It needs enough trials of each class, and it fails rather than falling back to an unstratified split. It keeps
approximately the proportions of trials, not of windows, does not balance classes and loses the temporal reading of
design A.

**Same call for both designs:**

```python
result = loso_calibrated(X, meta["subject"].to_numpy(), meta["label"].to_numpy(), partition, TSALR(),
                         segment_ids=meta["segment"].to_numpy(), window_index=meta["position"].to_numpy(),
                         window_ids=window_ids)
print(f"Mean test accuracy: {result.mean_accuracy:.1%}")
```

Symmetric LGC was applied to whole segments; since the partition never splits a segment, this equals applying it to
the calibration and test parts separately. Random LGC (`lgc_random(..., seed=42)`) needs the target prepared
separately and passed as `target_covs` ([examples/team_metrics_calibrated_tsa.py](examples/team_metrics_calibrated_tsa.py)).
Worked examples on Team Metrics and DEAP (data not included): [examples/](examples/README.md).

## Results

Mean accuracy (%) ± standard deviation across subjects, leave-one-subject-out, every subject as target. Windows of
4 s with a 2 s step at 128 Hz, Ledoit-Wolf covariances, LGC K = 10. **Team Metrics** (TNO): mental workload,
34 participants, binary. **DEAP** [4]: valence or arousal (rating ≥ 5), 32 participants, 15–36 Hz.

Earlier evaluations were affected by target-label leakage through FgMDM's internal discriminant fitting and, in DEAP
with LGC, segment boundaries inferred from target labels. These issues have been corrected by using MDM and
recording-based segment boundaries. The corrected results are reported below.

### Target without labels — corrected results

| Method | Team Metrics, alpha | Team Metrics, theta | DEAP, valence | DEAP, arousal |
|---|:---:|:---:|:---:|:---:|
| RCT + MDM | 53.44 ± 4.52 | 51.89 ± 2.70 | 54.19 ± 8.44 | 50.53 ± 6.05 |
| LGC (symmetric) + RCT + MDM | **56.85 ± 10.19** | **54.61 ± 6.97** | **55.70 ± 9.89** | **51.00 ± 6.55** |

Highest mean per column in bold, within each table.

The Team Metrics figures also include a correction of the filter sampling rate (128 Hz); their change with respect to
the published values is not attributable to the leakage alone. Published values and the scope of each correction:
[CORRECTIONS.md](CORRECTIONS.md).

### Additional configurations

These are not corrections of published figures. TSA-LR uses labelled calibration windows of the target and is
therefore not comparable with the unlabelled-target rows above as if it had the same information.

| Method | Target data | Team Metrics, alpha | DEAP, valence | DEAP, arousal |
|---|---|:---:|:---:|:---:|
| LGC (random, seed 42) + RCT + MDM | without labels | 60.71 ± 15.49 | 56.21 ± 10.77 | 51.36 ± 7.16 |
| TSA-LR | calibration | 56.88 ± 7.16 | 58.80 ± 9.15 | 58.30 ± 11.09 |
| LGC (symmetric) + TSA-LR | calibration | **63.41 ± 15.69** | 62.54 ± 13.86 | **60.91 ± 13.83** |
| LGC (random, seed 42) + TSA-LR | calibration | — | **63.37 ± 15.06** | 59.34 ± 11.41 |

Calibration partitions: Team Metrics, recorded blocks 1–6 calibrate and blocks 7–8 are the test; DEAP, a quarter of
each class's trials of the target are the test (stratified by class with seed 20261012, built once before evaluation).
Accuracy is computed on the test windows. Differences between means are descriptive; no significance test is reported
here.

The package reproduces the predictions behind every figure in these tables window by window in the registered
environment ([VALIDATION.md](VALIDATION.md)).

## IEEE MetroXRAINE 2026 talk
<!-- Fixed section: target of the QR code shown in the talk. Do not rename this heading. -->
Corrected DEAP results for RCT + MDM and LGC + RCT + MDM (valence and arousal) are
available in [Target without labels — corrected results](#target-without-labels--corrected-results).

To compute this kind of result with the toolkit: covariances per DEAP trial (each trial is a recorded segment),
optional `lgc_symmetric` with `half_window=10`, then `RCTMDM` evaluated with `loso_unlabelled_target`.
[examples/deap_stratified_split.py](examples/deap_stratified_split.py) shows how to load DEAP and build the
covariances and metadata; it runs the calibrated TSA-LR route, so it is a usage example, not an exact reproduction of
the table.

## Reproducibility

- The method is deterministic given the data, except random LGC (seed per call) and TSA-LR (fit seed, default 42),
  whose randomness is controlled and documented.
- Exact per-window agreement with the reference results is claimed only in the registered environment (Python
  3.9.16, NumPy 1.26.4, PyRiemann 0.9, scikit-learn 1.4.2); other versions may change predictions near decision
  boundaries.
- Team Metrics is a private dataset and is not redistributed; DEAP is available from its authors and is not
  redistributed either.

Environment, seeds, tests and data: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Validated configurations
and limitations: [VALIDATION.md](VALIDATION.md).

## Dataset availability

The Team Metrics dataset is proprietary (TNO, Netherlands) and cannot be shared. To request access for replication of
the reported results, contact Anne-Marie Brouwer (TNO,
[anne-marie.brouwer@tno.nl](mailto:anne-marie.brouwer@tno.nl)).

The DEAP example ([examples/deap_stratified_split.py](examples/deap_stratified_split.py)) runs on the publicly
available DEAP dataset, which is not redistributed here. See [examples/README.md](examples/README.md#input-formats)
for download instructions.

## Repository structure

```
lgc-toolkit/
├── src/lgctk/                         # Installable Python package (import lgctk)
│   ├── lgc.py                         # LGC: symmetric and random neighbourhoods within recorded segments
│   ├── pipelines.py                   # RCTMDM (unlabelled target) and TSALR (labelled calibration)
│   ├── evaluation.py                  # Leave-one-subject-out evaluators; the only holders of evaluation labels
│   ├── validation.py                  # Checks of subject/segment metadata and calibration/test partitions
│   ├── preprocessing.py               # Optional: band-pass, sliding windows, Ledoit–Wolf covariances
│   └── leakage.py                     # Helpers for the evaluation-label invariance test
├── examples/
│   ├── csv_quickstart.py              # Your own CSV: both evaluation routes (start here)
│   ├── team_metrics_transductive.py   # Team Metrics (private dataset): LGC + RCT + MDM
│   ├── team_metrics_calibrated_tsa.py # Team Metrics (private dataset): LGC + TSA-LR
│   ├── deap_stratified_split.py       # DEAP (request from its authors): LGC + TSA-LR
│   └── README.md                      # Input formats, parameters and the scientist's decisions
├── tests/                             # Synthetic data only: LGC, metadata, label firewall, invariance
├── docs/
│   ├── METHOD.md                      # Formulas and parameters
│   ├── PROTOCOLS.md                   # Data access per protocol, segments, limitations
│   └── REPRODUCIBILITY.md             # Environment, seeds, tests and data
├── CORRECTIONS.md                     # Corrections of results obtained with lgc-rct
├── VALIDATION.md                      # Validated configurations and their evidence
├── CHANGELOG.md
├── CITATION.cff
└── pyproject.toml
```

## Citation

If you use lgc-toolkit, please cite the software version used in your experiments (metadata in
[CITATION.cff](CITATION.cff)). A Zenodo DOI will be added with the first archived release.
<!-- Zenodo: after the deposit, replace the sentence above with the concept DOI (all versions) and add
     doi = {10.5281/zenodo.<version id>} and publisher = {Zenodo} to the entry below. -->

```bibtex
@software{licona2026lgctoolkit,
  author  = {Licona Mu{\~n}oz, Ricardo and Guetschel, Pierre and Bruin, Juliette and
             Yilmaz, Efecan and Brouwer, Anne-Marie},
  title   = {lgc-toolkit: Local Geometric Consistency for cross-subject {EEG} transfer learning},
  year    = {2026},
  version = {0.1.0.dev0},
  url     = {https://github.com/RickLicona/lgc-toolkit}
}
```

Earlier publications introducing LGC are listed below; corrections to their reported results are documented in
[CORRECTIONS.md](CORRECTIONS.md).

```bibtex
@inproceedings{licona2026lgc,
  title     = {Local Geometric Consistency for Cross-Subject {EEG} Transfer Learning},
  author    = {Licona Mu{\~n}oz, Ricardo and Guetschel, Pierre and Bruin, Juliette and
               Yilmaz, Efecan and Brouwer, Anne-Marie},
  booktitle = {Proceedings of the Fifth Neuroadaptive Technology Conference (NAT'26)},
  year      = {2026},
  note      = {Berlin, Germany}
}
```

## References

[1] Zanini, P., Congedo, M., Jutten, C., Said, S., & Berthoumieu, Y. (2018). Transfer learning: A Riemannian
geometry framework with applications to brain–computer interfaces. *IEEE Transactions on Biomedical Engineering*,
65(5), 1107–1116. https://doi.org/10.1109/TBME.2017.2742541

[2] Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012). Multiclass brain–computer interface classification
by Riemannian geometry. *IEEE Transactions on Biomedical Engineering*, 59(4), 920–928.
https://doi.org/10.1109/TBME.2011.2172210

[3] Bleuzé, A., Mattout, J., & Congedo, M. (2022). Tangent space alignment: Transfer learning for brain-computer
interface. *Frontiers in Human Neuroscience*, 16, 1049985. https://doi.org/10.3389/fnhum.2022.1049985

[4] Koelstra, S., Muhl, C., Soleymani, M., Lee, J.-S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., & Patras, I.
(2012). DEAP: A database for emotion analysis using physiological signals. *IEEE Transactions on Affective
Computing*, 3(1), 18–31. https://doi.org/10.1109/T-AFFC.2011.15

This repository supersedes [lgc-rct](https://github.com/RickLicona/lgc-rct) for new work.

## License

MIT License — see [LICENSE](LICENSE).
