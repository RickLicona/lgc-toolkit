# Reproducibility

## Data
- **Team Metrics** (mental workload, 34 participants): **private dataset, not publicly available and not
  redistributed.** [examples/](../examples/README.md) document the expected input format; results on it can only be
  obtained with authorised access to the data.
- **DEAP** (emotion, 32 participants): available on request from its authors under their licence; it is **not
  redistributed** here. The DEAP example expects the preprocessed MATLAB release (`data_preprocessed_matlab`:
  `s01.mat` … `s32.mat`), stored locally by the user.

## Installation
```bash
git clone https://github.com/RickLicona/lgc-toolkit
cd lgc-toolkit
pip install ".[test]" pandas      # pandas is only needed by the examples
python -m pytest
```

## Tests
The suite (`tests/`, synthetic data only, about three minutes) covers:
- **LGC.** Symmetric and random neighbourhoods and means equal to a reference implementation of the algorithm
  (symmetric: K = 1, 3, 10; random: K = 1, 3, 10, 20 and seeds 42, 43, 44), segment boundaries, seed handling and
  input validation.
- **Metadata and partitions.** Order, composite subject/segment identity, partitions split inside a segment, targets
  without a calibration or a test side.
- **Label firewall.** Each pipeline and evaluator receives only the data its protocol allows; in the calibrated
  protocol, the rows reaching every fitted stage are recorded and no test window may reach a fit.
- **Evaluation-label invariance.** For RCT + MDM and TSA-LR, with and without symmetric or random LGC, the target's
  evaluation labels are replaced by fake labels at the source (inversion, constants, random per segment and per
  window) and the whole path is re-run: predictions, LGC representations and fitted state must not change. Positive
  controls with a known dependence on the labels (an FgMDM fit with weight-0 target labels, segment boundaries
  inferred from labels, test windows added to the calibration) must be detected, so that a pass is informative.
- **Examples.** The Team Metrics feature extraction and the CSV quick start (window counts, partitions constant per
  segment, acquisition order taken from the file, no fallback to an unstratified split, input checks).

A pass is evidence for the tested cases and versions; it does not exclude every leakage route (e.g. duplicated data
across partitions, metadata that encode the class, or class information carried by the segment structure; see
[PROTOCOLS.md](PROTOCOLS.md#limitations)).

## Validated figures
The figures in [VALIDATION.md](../VALIDATION.md) come from runs of this package compared, window by window, with the
reference pipelines that produced the results; the comparison scripts read those (private) references and are not
part of this repository.

## Environment
Reference environment for the exact per-window parity checks: Python 3.9.16, NumPy 1.26.4, pandas 2.2.2,
PyRiemann 0.9, scikit-learn 1.4.2 (SciPy 1.13.1, macOS arm64, OpenBLAS). Bit-identical predictions are only claimed in
this environment; other platforms or library versions may change predictions near decision boundaries.

PyRiemann 0.9 as published on PyPI declares `numpy>=2.0`, so a plain `pip install` resolves NumPy 2. The registered
environment combines it with NumPy 1.26.4; pip reports that combination as a conflict, and it is only obtained by
pinning NumPy after installing. The test suite passes in both: NumPy 1.26.4 with scikit-learn 1.4.2, and NumPy 2.0.2
with scikit-learn 1.4.2 and 1.6.1 (Python 3.9.16). Bit-identical parity is claimed only for the registered
environment.

Numerical details kept for parity: Team Metrics fits of `loso_unlabelled_target` use the source subjects followed by
the target; DEAP fits use the subjects in numerical order with BLAS limited to one thread (`fit_order`,
`RCTMDM(blas_threads=1)`). Both orders are valid; they mirror the audited producers.

## Randomness
- **Random LGC.** `lgc_random` creates a fresh `numpy.random.default_rng(seed)` per subject and per call. The draws
  therefore depend on which windows enter the call: LGC over a whole subject and LGC applied separately to its
  calibration and test parts give different random neighbourhoods (they never leave the segment in either case).
  Record the scope together with the seed.
- **TSA-LR.** `TLRotate` fits a one-component PCA without `random_state`, and scikit-learn may use its randomized
  solver, which draws from the legacy global NumPy RNG. `TSALR` seeds that RNG with `fit_seed` (default 42) during the
  fit, restores the previous state afterwards and, by default, limits BLAS to one thread during fit and predict
  (`blas_threads=1`). Do not run `TSALR` fits concurrently in threads of one process: they share the global RNG. The
  validated parity runs also set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
  `VECLIB_MAXIMUM_THREADS` to 1 before importing NumPy, as the reference producers did.
- **Partitions.** A seeded partition (e.g. stratified by class) is built once and frozen; its seed is separate from
  the LGC seed and the TSA fit seed.
