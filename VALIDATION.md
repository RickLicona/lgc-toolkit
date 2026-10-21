# Validation

How the results reported in the [README](README.md#results) are tied to this package.

## Evidence
1. **Exact per-window agreement on real data.** For every configuration below, the package was run with every subject
   as target and all other subjects as sources, and its predictions were compared by window identifier with those of
   the reference pipelines that produced the results: identical `y_true` and `y_pred`, identical sets of windows, no
   duplicates. Total: 514 864 window predictions compared over all configurations, 0 differences.
2. **Synthetic tests in the package** (`tests/`): LGC against a reference implementation, the label firewall and the
   evaluation-label invariance test with positive controls ([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md#tests)).
   They test the mechanism of the package; they are not an invariance run on each real configuration.

The comparison scripts read the private reference results and are not part of this repository.

## Configurations
Windows of 4 s, step 2 s, 128 Hz, 32 channels, Ledoit-Wolf covariances; LGC K = 10; random LGC with seed 42.

| Configuration | Dataset, band, task | Target data | Subjects | Windows compared | Also identical |
|---|---|---|:---:|:---:|---|
| RCT + MDM | Team Metrics, alpha 8–13 Hz | without labels | 34 | 40 528 | — |
| LGC (symmetric) + RCT + MDM | Team Metrics, alpha | without labels | 34 | 40 528 | — |
| RCT + MDM | Team Metrics, theta 4–7 Hz | without labels | 34 | 40 528 | — |
| LGC (symmetric) + RCT + MDM | Team Metrics, theta | without labels | 34 | 40 528 | — |
| LGC (random) + RCT + MDM | Team Metrics, alpha | without labels | 34 | 40 528 | MDM class means |
| RCT + MDM | DEAP, 15–36 Hz, valence | without labels | 32 | 38 400 | — |
| LGC (symmetric) + RCT + MDM | DEAP, valence | without labels | 32 | 38 400 | — |
| LGC (random) + RCT + MDM | DEAP, valence | without labels | 32 | 38 400 | — |
| RCT + MDM | DEAP, arousal | without labels | 32 | 38 400 | — |
| LGC (symmetric) + RCT + MDM | DEAP, arousal | without labels | 32 | 38 400 | — |
| LGC (random) + RCT + MDM | DEAP, arousal | without labels | 32 | 38 400 | — |
| TSA-LR | Team Metrics, alpha; blocks 1–6 / 7–8 | calibration | 34 | 10 132 | calibration windows; LR coefficients |
| LGC (symmetric) + TSA-LR | Team Metrics, alpha; blocks 1–6 / 7–8 | calibration | 34 | 10 132 | calibration windows; LR coefficients |
| TSA-LR | DEAP, valence; stratified split | calibration | 32 | 10 290 | calibration/test windows |
| LGC (symmetric) + TSA-LR | DEAP, valence; stratified split | calibration | 32 | 10 290 | calibration/test windows |
| LGC (random) + TSA-LR | DEAP, valence; stratified split | calibration | 32 | 10 290 | calibration/test windows |
| TSA-LR | DEAP, arousal; stratified split | calibration | 32 | 10 230 | calibration/test windows |
| LGC (symmetric) + TSA-LR | DEAP, arousal; stratified split | calibration | 32 | 10 230 | calibration/test windows |
| LGC (random) + TSA-LR | DEAP, arousal; stratified split | calibration | 32 | 10 230 | calibration/test windows |

In the calibrated rows, windows compared are the target's test windows; the target's LGC is applied separately to its
calibration and test parts, and the sources use LGC over the whole subject. The DEAP split puts a quarter of each
class's trials of the target in the test (seed 20261012), built once before evaluation
([docs/PROTOCOLS.md](docs/PROTOCOLS.md)).

**Random neighbourhoods.** For seeds 42, 43 and 44, the ordered list of neighbours of every window equals the
reference: in Team Metrics, the neighbourhoods stored with the reference results (3 × 40 528 windows); in DEAP, the
reference implementation of the algorithm run on the same windows, whole subjects and each task's calibration and test
parts (3 × 115 200 windows). Only seed 42 has a full predictive comparison.

## Environment
Python 3.9.16, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, PyRiemann 0.9, scikit-learn 1.4.2 (macOS arm64, OpenBLAS).
Exact agreement is claimed only in this environment ([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md#environment)).

## Provenance
The runs of the random-LGC and TSA-LR configurations recorded, at start and at the end, the hashes of the imported
package, the comparison scripts, the references, every input file, the configuration and the environment, and
nothing changed during the runs. The earlier runs of the RCT + MDM and symmetric LGC + RCT + MDM configurations
recorded the hashes of their scripts and outputs but not of the imported package or the input data: they prove the
integrity of the outputs, not a cryptographic link to a given version of the source code or the inputs.

## Scope
Agreement with the reference pipelines shows that this package reproduces them; it does not validate other datasets,
parameters or library versions. Protocol limitations (offline use, block structure, DEAP stimuli):
[docs/PROTOCOLS.md](docs/PROTOCOLS.md#limitations). Team Metrics is private; DEAP is not redistributed.
