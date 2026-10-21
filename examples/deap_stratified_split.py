"""Example 3 -- DEAP, stratified per-trial calibration/test split: [LGC] + TSA-LR, leave-one-subject-out.

DEAP is not redistributed. Request it from its authors (https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), download the
PREPROCESSED MATLAB release (folder ``data_preprocessed_matlab``: s01.mat ... s32.mat; ``data`` 40 trials x 40
channels x 8064 samples at 128 Hz, the first 32 channels EEG; ``labels`` 40 x 4 ratings), keep it locally and pass
``--data /path/to/data_preprocessed_matlab``. This example reads the files with ``scipy.io.loadmat``; it does not
read the Python (pickle) release.

WHAT THE SCIENTIST DECIDES (this example shows the recommended way; it is NOT a package function):

1. Segments. ``segment_ids`` = the trial index of the file (acquisition metadata). LGC neighbourhoods never leave a
   trial, so they never cross from one trial to another.
2. The calibration/test split, per subject and task, built ONCE, before any evaluation (benchmark construction). The
   rule here: per class, ceil(n_c / 4) trials go to test, drawn with ``np.random.default_rng(split_seed)``; the rest
   calibrate. Stratifying uses the per-trial labels of the target, test trials included: that is a declared design
   decision of the benchmark, made outside the predictive function, which never receives test labels. The split is
   then FROZEN: an invariance test must not recompute it when it injects fake labels. ``split_seed`` is not the LGC
   seed nor the TSA fit seed.
3. LGC scope. Sources: LGC over the whole subject. Target: LGC applied SEPARATELY to its calibration trials and to its
   test trials, so that no neighbourhood mixes calibration with test. ``loso_calibrated`` receives the target
   representation through ``target_covs``. LGC of the test part uses test covariances of the same trial (offline
   preprocessing, not a fit).

Labels: rating >= 5 -> class 1. Per trial (one segment at a time, so no window crosses a trial): zero-phase
Butterworth band-pass 15-36 Hz of order 4 (``lgctk.preprocessing.bandpass``), sliding windows (``sliding_windows``;
validated: 4 s every 2 s = 30 windows, the 3 s pre-stimulus baseline kept), Ledoit-Wolf covariances
(``lw_covariances``). Window length and step are arguments; a different choice does not inherit validated figures.

    python examples/deap_stratified_split.py --data /path/to/data_preprocessed_matlab --task valence
    python examples/deap_stratified_split.py --data ... --task arousal --lgc random --seed 42 --split-seed 20261012
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from lgctk import TSALR, lgc_random, lgc_symmetric, loso_calibrated
from lgctk.preprocessing import bandpass, lw_covariances, sliding_windows

BAND, FS, WINDOW_SEC, STEP_SEC = (15, 36), 128, 4, 2
N_TRIALS, N_EEG = 40, 32
TASKS = {"valence": 0, "arousal": 1}


def load_subject(path):
    mat = scipy.io.loadmat(str(path))
    return np.array(mat["data"])[:, :N_EEG, :], np.array(mat["labels"])


def subject_features(X, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    win, step = int(round(window_sec * FS)), int(round(step_sec * FS))
    covs, trial, pos = [], [], []
    for t in range(N_TRIALS):
        w, _ = sliding_windows(bandpass(X[t], BAND[0], BAND[1], fs=FS, axis=1), win, step)
        covs.append(lw_covariances(w))
        trial += [t] * len(w)
        pos += list(range(len(w)))
    return np.concatenate(covs), np.array(trial), np.array(pos)


def stratified_split(y_trials, split_seed):
    """Benchmark construction (scientist's choice): per class, ceil(n_c / 4) trials to test. Returns the test trials,
    or None when a class has fewer than two trials (the subject is then not evaluable for this task)."""
    rng = np.random.default_rng(split_seed)
    test = []
    for c in (0, 1):
        ids = np.sort(np.flatnonzero(y_trials == c))
        if len(ids) < 2:
            return None
        test += rng.permutation(ids)[:math.ceil(len(ids) / 4)].tolist()
    return np.array(sorted(test), dtype=int)


def apply_lgc(covs, subj, seg, pos, kind, k, seed):
    if kind == "symmetric":
        return lgc_symmetric(covs, subj, seg, pos, half_window=k)
    if kind == "random":
        return lgc_random(covs, subj, seg, pos, half_window=k, seed=seed)
    return covs.copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="folder with DEAP s01.mat ... s32.mat")
    ap.add_argument("--task", choices=list(TASKS), default="valence")
    ap.add_argument("--lgc", choices=["symmetric", "random", "none"], default="symmetric")
    ap.add_argument("--half-window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="seed of random LGC")
    ap.add_argument("--split-seed", type=int, default=20261012, help="seed of the calibration/test split")
    ap.add_argument("--window-sec", type=float, default=4.0, help="window length in seconds (validated: 4)")
    ap.add_argument("--step-sec", type=float, default=2.0, help="window step in seconds (validated: 2, i.e. 50%% overlap)")
    a = ap.parse_args()

    files = sorted(Path(a.data).glob("s*.mat"))
    covs, subj, seg, pos, y, partition = [], [], [], [], [], []
    for f in files:
        X, ratings = load_subject(f)
        y_trials = (ratings[:, TASKS[a.task]] >= 5.0).astype(int)
        test_trials = stratified_split(y_trials, a.split_seed)       # 1. benchmark: built once, then frozen
        if test_trials is None:
            print(f"  {f.stem}: skipped for {a.task} (a class has fewer than two trials)")
            continue
        C, tr, ps = subject_features(X, a.window_sec, a.step_sec)
        covs.append(C)
        subj += [f"subject_{int(f.stem[1:]):02d}"] * len(tr)
        seg.append(tr)
        pos.append(ps)
        y.append(y_trials[tr])
        partition.append(np.where(np.isin(tr, test_trials), "test", "calibration"))
        print(f"  covariances {f.stem}", flush=True)
    covs, subj = np.concatenate(covs), np.array(subj)
    seg, pos, y, partition = map(np.concatenate, (seg, pos, y, partition))

    X_source = apply_lgc(covs, subj, seg, pos, a.lgc, a.half_window, a.seed)          # 3. sources: whole subject
    X_target = np.empty_like(X_source)
    for part in ("calibration", "test"):                                               # 3. target: per partition
        m = partition == part
        X_target[m] = apply_lgc(covs[m], subj[m], seg[m], pos[m], a.lgc, a.half_window, a.seed)

    window_ids = np.array([f"t{t:02d}_w{p:02d}" for t, p in zip(seg, pos)])
    res = loso_calibrated(X_source, subj, y, partition, TSALR(), segment_ids=seg, window_index=pos,
                          window_ids=window_ids, target_covs=X_target)
    print(pd.DataFrame(res.per_subject).to_string(index=False))
    print(f"{a.task}: mean test accuracy {100 * res.mean_accuracy:.2f}% over {len(res.per_subject)} targets")


if __name__ == "__main__":
    main()
