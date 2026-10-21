"""Quick start -- your own CSV: covariances, LGC and the two leave-one-subject-out routes (not part of the package API).

Input: one CSV, one row per SAMPLE, in temporal order, with the columns

    subject, segment, label, <channel>, <channel>, ...

``subject`` identifies the participant; ``segment`` is a recorded trial or block, unique within the subject (it may
repeat across subjects); ``label`` is the condition, one integer per segment with the same meaning across subjects;
the channel columns are numeric EEG amplitudes in consistent units. The rows of each segment are contiguous and every
subject occupies one contiguous run. ``--fs`` must be the real sampling rate of the file: it does not resample.

What belongs to this example and what to the package:

- example (this file): ``make_synthetic_csv``, ``prepare`` (input checks, then per segment: band-pass, sliding
  windows, covariances, window positions and IDs), ``partition_by_acquisition_order`` and ``partition_stratified``;
- package (``lgctk``): ``bandpass``, ``sliding_windows``, ``lw_covariances``, ``lgc_symmetric``, ``RCTMDM``, ``TSALR``,
  ``loso_unlabelled_target`` and ``loso_calibrated``.

Routes:

- ``unlabelled``: each participant in turn is the target; its covariances, without labels, enter the re-centring of
  RCT + MDM and all its windows are predicted and scored.
- ``acquisition-order``: TSA-LR; the first ``--calibration-segments`` segments of each target, in the order they
  appear in the file (acquisition order), calibrate with labels; the rest are the held-out test.
- ``stratified``: TSA-LR; about 25 % of the segments of each target go to test, stratified by class, seed 20261012.

Symmetric LGC is applied to whole segments; since the partition is constant inside every segment, this equals
applying it to the calibration and test parts separately. Random LGC needs that separation explicitly
(``target_covs``; see ``team_metrics_calibrated_tsa.py``).

    python examples/csv_quickstart.py --make-synthetic eeg_dataset.csv
    python examples/csv_quickstart.py --data eeg_dataset.csv --channels Fp1 Fp2 --fs 128
    python examples/csv_quickstart.py --data eeg_dataset.csv --channels Fp1 Fp2 --fs 128 --route stratified
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lgctk import RCTMDM, TSALR, lgc_symmetric, loso_calibrated, loso_unlabelled_target
from lgctk.preprocessing import bandpass, lw_covariances, sliding_windows

KEYS = ["subject", "segment", "label"]
ROUTES = ["unlabelled", "acquisition-order", "stratified"]


def make_synthetic_csv(path, n_subjects=3, n_segments=8, segment_sec=30, fs=128, seed=0):
    """Write a synthetic recording in the expected format: channels Fp1/Fp2, alternating classes. Not real EEG."""
    rng = np.random.default_rng(seed)
    t = np.arange(segment_sec * fs) / fs
    frames = []
    for s in range(1, n_subjects + 1):
        gain = rng.uniform(0.5, 2.0)                                   # subject-specific scale (domain shift)
        for k in range(1, n_segments + 1):
            label = (k - 1) % 2
            alpha = (1 + 0.1 * label) * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            eeg = gain * (rng.standard_normal((2, len(t))) + np.array([[1.0], [0.5]]) * alpha)
            frames.append(pd.DataFrame({"subject": f"S{s:02d}", "segment": f"trial_{k:02d}", "label": label,
                                        "Fp1": eeg[0].round(4), "Fp2": eeg[1].round(4)}))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _check_input(data, channels, fs, band):
    if data.empty:
        raise ValueError("the CSV has no rows")
    missing = [c for c in KEYS + list(channels) if c not in data.columns]
    if missing:
        raise ValueError(f"the CSV lacks columns {missing}")
    for c in KEYS:
        if data[c].isna().any():
            raise ValueError(f"column {c!r} has missing values in rows {data.index[data[c].isna()][:5].tolist()}")
    for c in channels:
        if not pd.api.types.is_numeric_dtype(data[c]):
            raise ValueError(f"channel {c!r} is not numeric")
        if not np.isfinite(data[c].to_numpy(dtype=float)).all():
            raise ValueError(f"channel {c!r} has non-finite values")
    labels = data["label"]
    if not pd.api.types.is_numeric_dtype(labels) or not (labels == labels.round()).all():
        raise ValueError("labels must be integers (e.g. 0/1)")
    if not 0 < band[0] < band[1] < fs / 2:
        raise ValueError(f"band {band} Hz must lie between 0 and the Nyquist frequency {fs / 2} Hz")
    key = list(zip(data["subject"], data["segment"]))
    starts = [i for i in range(len(key)) if i == 0 or key[i] != key[i - 1]]
    if len({key[i] for i in starts}) != len(starts):
        raise ValueError("the rows of each (subject, segment) must be contiguous")
    subj = data["subject"].to_numpy()
    subject_starts = np.flatnonzero(np.r_[True, subj[1:] != subj[:-1]])
    if len(set(subj[subject_starts].tolist())) != len(subject_starts):
        raise ValueError("the rows of each subject must be contiguous")


def prepare(data, channels, fs, band=(8, 13), window_sec=4, step_sec=2, half_window=10):
    """Per segment: band-pass, sliding windows, Ledoit-Wolf covariances; then symmetric LGC.

    Returns ``X`` (LGC covariances), ``meta`` (one row per window: subject, segment, position, label, in data order)
    and ``window_ids`` (unique integers). Labels are carried along for the evaluator; LGC does not use them.
    """
    _check_input(data, channels, fs, band)
    covariances, metadata = [], []
    for (subject, segment), rows in data.groupby(["subject", "segment"], sort=False):
        if rows["label"].nunique() != 1:
            raise ValueError(f"segment {(subject, segment)} has more than one label")
        eeg = rows[channels].to_numpy(dtype=float).T                 # (channels, samples)
        filtered = bandpass(eeg, lo=band[0], hi=band[1], fs=fs, axis=1, order=4)
        try:
            windows, starts = sliding_windows(filtered, window_len=round(window_sec * fs),
                                              step_len=round(step_sec * fs))
        except ValueError as err:
            raise ValueError(f"segment {(subject, segment)}: {err}") from None
        covariances.append(lw_covariances(windows))
        label = int(rows["label"].iloc[0])
        metadata.extend((subject, segment, position, label) for position in range(len(windows)))
    covs = np.concatenate(covariances)
    meta = pd.DataFrame(metadata, columns=["subject", "segment", "position", "label"])
    window_ids = np.arange(len(meta))
    X = lgc_symmetric(covs, meta["subject"].to_numpy(), meta["segment"].to_numpy(), meta["position"].to_numpy(),
                      half_window=half_window)
    return X, meta, window_ids


def _per_window(meta, assignment):
    return np.array([assignment[key] for key in zip(meta["subject"], meta["segment"])])


def partition_by_acquisition_order(meta, n_calibration=6):
    """The first ``n_calibration`` segments of each subject, in the order they appear in the data (which must be the
    acquisition order; segment names are never sorted), calibrate; the others are the test."""
    assignment = {}
    for subject, rows in meta.groupby("subject", sort=False):
        order = rows["segment"].unique()                              # order of first appearance
        if not 0 < n_calibration < len(order):
            raise ValueError(f"subject {subject!r} has {len(order)} segments; cannot calibrate on {n_calibration}")
        for i, segment in enumerate(order):
            assignment[(subject, segment)] = "calibration" if i < n_calibration else "test"
        calib_labels = rows.loc[rows["segment"].isin(order[:n_calibration]), "label"]
        if calib_labels.nunique() < meta["label"].nunique():
            raise ValueError(f"the calibration segments of subject {subject!r} do not contain every class")
    return _per_window(meta, assignment)


def partition_stratified(meta, test_size=0.25, seed=20261012):
    """About ``test_size`` of the segments of each subject go to test, stratified by class. Splits SEGMENTS (never
    windows) and fails instead of falling back to a non-stratified split."""
    segments = meta[["subject", "segment", "label"]].drop_duplicates()
    assignment = {}
    for subject, rows in segments.groupby("subject", sort=False):
        try:
            calibration_ids, test_ids = train_test_split(rows["segment"].to_numpy(), test_size=test_size,
                                                         stratify=rows["label"].to_numpy(), random_state=seed)
        except ValueError as err:
            raise ValueError(f"subject {subject!r}: cannot build a stratified split ({err})") from None
        for segment in calibration_ids:
            assignment[(subject, segment)] = "calibration"
        for segment in test_ids:
            assignment[(subject, segment)] = "test"
    return _per_window(meta, assignment)


def run(X, meta, window_ids, route, n_calibration=6):
    subjects, labels = meta["subject"].to_numpy(), meta["label"].to_numpy()
    if route == "unlabelled":
        return loso_unlabelled_target(X, subjects, labels, RCTMDM(), window_ids=window_ids)
    if route == "acquisition-order":
        partition = partition_by_acquisition_order(meta, n_calibration)
    else:
        partition = partition_stratified(meta)
    return loso_calibrated(X, subjects, labels, partition, TSALR(), segment_ids=meta["segment"].to_numpy(),
                           window_index=meta["position"].to_numpy(), window_ids=window_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV with columns subject, segment, label and the EEG channels")
    src.add_argument("--make-synthetic", metavar="PATH", help="write a synthetic CSV in the expected format and exit")
    ap.add_argument("--channels", nargs="+", help="channel columns to use, in a fixed order")
    ap.add_argument("--fs", type=float, help="real sampling rate of the CSV in Hz (no resampling is done)")
    ap.add_argument("--band", nargs=2, type=float, default=[8, 13], help="band-pass edges in Hz")
    ap.add_argument("--window-sec", type=float, default=4.0)
    ap.add_argument("--step-sec", type=float, default=2.0)
    ap.add_argument("--half-window", type=int, default=10, help="symmetric LGC radius K, in windows")
    ap.add_argument("--route", choices=ROUTES + ["all"], default="all")
    ap.add_argument("--calibration-segments", type=int, default=6, help="acquisition-order route only")
    a = ap.parse_args()

    if a.make_synthetic:
        make_synthetic_csv(a.make_synthetic)
        print(f"wrote {a.make_synthetic}")
        return
    if not a.channels or a.fs is None:
        ap.error("--data needs --channels and --fs")
    X, meta, window_ids = prepare(pd.read_csv(a.data), a.channels, a.fs, tuple(a.band), a.window_sec, a.step_sec,
                                  a.half_window)
    print(f"{len(meta)} windows, {meta['subject'].nunique()} subjects")
    for route in ROUTES if a.route == "all" else [a.route]:
        res = run(X, meta, window_ids, route, a.calibration_segments)
        print(f"{route}: mean accuracy {res.mean_accuracy:.1%} over {len(res.per_subject)} targets, "
              f"{len(res.y_true)} windows scored")


if __name__ == "__main__":
    main()
