# Changelog

## 0.1.0 (2026-10-21)
First release of `lgc-toolkit` (import `lgctk`), superseding `lgc-rct` for new work.

- **LGC.** Symmetric neighbourhoods (`lgc_symmetric`, `symmetric_neighbourhoods`) and random neighbourhoods
  (`lgc_random`, `random_neighbourhoods`; mandatory seed, fresh generator per subject and call). Neighbourhoods never
  cross a recording segment; subject, segment and window-order metadata are mandatory and validated
  (`lgctk.validation`); data are never reordered; integer covariances are promoted to float64.
- **Pipelines.** `RCTMDM`: Riemannian re-centring with an unlabelled target and MDM fitted on the sources (FgMDM,
  which ignores sample weights in its discriminant step, is not used). `TSALR`: tangent space alignment with a
  labelled target calibration and logistic regression; test covariances enter no fit; fit seed and BLAS threads
  controlled.
- **Evaluation.** Leave-one-subject-out evaluators that alone hold the evaluation labels: `loso_unlabelled_target`
  and `loso_calibrated` (calibration/test partition per recorded segment, validated by `check_partition` before any
  fit; separate LGC of the target's parts through `target_covs`).
- **Preprocessing utilities** (optional): zero-phase Butterworth band-pass, sliding windows, Ledoit-Wolf covariances.
- **Testing utilities** (`lgctk.leakage`): fake-label schemes and recorders of the data reaching each fitted stage.
- **Tests** on synthetic data: LGC against a reference implementation, metadata and partition validation, label
  firewall, evaluation-label invariance with positive controls.
- **Examples.** A quick start from a generic CSV file (one row per sample: subject, segment, label, channels) with
  both evaluation routes; Team Metrics (unlabelled target; calibrated TSA-LR) and DEAP (stratified calibration/test
  split).
- **Corrections** of results obtained with `lgc-rct`: [CORRECTIONS.md](CORRECTIONS.md).
