"""Signal preprocessing used by the reference pipelines.

Zero-phase Butterworth band-pass (order 4, ``filtfilt``) applied per recorded segment, sliding windows inside each
segment, and Ledoit-Wolf covariances. Windows never cross segment boundaries.
"""
import numpy as np
from pyriemann.estimation import Covariances
from scipy.signal import butter, filtfilt


def bandpass(signal, lo, hi, fs, axis, order=4):
    """Zero-phase Butterworth band-pass along ``axis``. ``fs`` is mandatory (no default sampling rate)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return filtfilt(b, a, signal, axis=axis)


def window_starts(n_samples, window_len, step_len):
    """Start samples of the sliding windows that fit entirely inside a segment of ``n_samples``."""
    return list(range(0, n_samples - window_len + 1, step_len))


def sliding_windows(signal, window_len, step_len):
    """Windows of a (n_channels, n_samples) segment -> (n_windows, n_channels, window_len) float32, and the starts."""
    starts = window_starts(signal.shape[1], window_len, step_len)
    if not starts:
        raise ValueError(f"segment of {signal.shape[1]} samples is shorter than one window ({window_len})")
    return np.stack([signal[:, s:s + window_len] for s in starts]).astype(np.float32), np.asarray(starts)


def lw_covariances(windows):
    """Ledoit-Wolf shrunk covariances of (n_windows, n_channels, n_times) windows."""
    return Covariances(estimator="lwf").transform(windows)
