"""Team Metrics loading and feature extraction shared by examples 1 and 2 (not part of the package API).

Team Metrics is a PRIVATE dataset: it is not distributed and these examples only run with authorised access. Expected
input: one CSV with one row per sample (128 Hz) and the columns

    Fp1, AF3, F7, F3, FC1, FC5, T7, C3, CP1, CP5, P7, P3, Pz, PO3, O1, Oz, O2, PO4, P4, P8, CP6, CP2, C4, T8,
    FC6, FC2, F4, F8, AF4, Fp2, Fz, Cz, time, block, class, subject

``block`` is the RECORDED block identifier (acquisition metadata) and ``class`` the condition of the block; the rows
of each block are in temporal order. Any dataset with the same shape (subject, recorded segment, label, channels)
can be adapted by changing the constants below.
"""
import numpy as np
import pandas as pd

from lgctk.preprocessing import bandpass, lw_covariances, sliding_windows

CHANNELS = ["Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
            "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8", "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz"]
BANDS = {"alpha": (8, 13), "theta": (4, 7)}
FS, WINDOW_SEC, STEP_SEC = 128, 4, 2


def domain(subject):
    return f"subject_{int(subject):02d}"


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in CHANNELS + ["block", "class", "subject"] if c not in df.columns]
    if missing:
        raise SystemExit(f"input CSV lacks columns {missing}")
    return df


def features(df, band, fs=FS, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    """Covariances and per-window metadata of every subject (sorted).

    Per recorded block: zero-phase Butterworth band-pass of order 4 (``lgctk.preprocessing.bandpass``), sliding
    windows of ``window_sec`` seconds every ``step_sec`` seconds inside the block (``sliding_windows``; in samples:
    ``window_sec * fs`` and ``step_sec * fs``; the defaults 4 s / 2 s overlap by 50 %), Ledoit-Wolf covariances
    (``lw_covariances``). Filtering and windowing are applied to one block at a time, so no window crosses a block.
    These are the settings of the validated configurations, not universal requirements: a different choice does not
    inherit the validated figures.
    """
    lo, hi = BANDS[band]
    win, step = int(round(window_sec * fs)), int(round(step_sec * fs))
    covs, meta = [], []
    for subject, df_subj in df.groupby("subject"):
        W = []
        for block in sorted(df_subj["block"].unique()):
            rows = df_subj[df_subj["block"] == block]
            label = int(rows["class"].iloc[0])
            sig = bandpass(rows[CHANNELS].to_numpy(), lo, hi, fs=fs, axis=0)       # (samples, channels)
            windows, starts = sliding_windows(sig.T, win, step)                        # (n, channels, win) float32
            W.append(windows)
            meta += [(domain(subject), int(block), k, int(start), label) for k, start in enumerate(starts)]
        covs.append(lw_covariances(np.concatenate(W)))
        print(f"  {band} covariances {domain(subject)}", flush=True)
    meta = pd.DataFrame(meta, columns=["subject", "block", "window_in_block", "start_sample", "y"])
    meta["window_id"] = meta["block"].map("b{}".format) + "_w" + meta["window_in_block"].map("{:04d}".format)
    return np.concatenate(covs), meta
