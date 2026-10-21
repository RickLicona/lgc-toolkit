"""Validation of per-window acquisition metadata.

Every window carries a subject identifier, a segment identifier (recorded block or trial) and a position within its
segment. The package never infers segments, never reorders data and refuses ambiguous metadata.
"""
import numpy as np


class MetadataError(ValueError):
    """Raised when per-window metadata are missing, inconsistent or ambiguous."""


def _as_1d(name, a, n):
    if a is None:
        raise MetadataError(f"{name} is required")
    a = np.asarray(a)
    if a.ndim != 1 or len(a) != n:
        raise MetadataError(f"{name} must be 1-D with {n} entries, got shape {a.shape}")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise MetadataError(f"{name} contains non-finite values")
    if a.dtype.kind == "O" and any(v is None or (isinstance(v, float) and v != v) for v in a):
        raise MetadataError(f"{name} contains missing values")
    return a


def segment_runs(subject_ids, segment_ids, window_index):
    """Validate metadata and return the contiguous segment runs.

    Parameters
    ----------
    subject_ids, segment_ids : array-like, shape (n_windows,)
        Segment identity is the composite key (subject, segment): the same segment label may appear in different
        subjects. Each subject must occupy one contiguous run and, inside it, each segment one contiguous run.
    window_index : array-like of int, shape (n_windows,)
        Position of each window within its segment (e.g. window number or start sample). Must be strictly increasing
        inside each segment: the data are expected in temporal order and are never reordered.

    Returns
    -------
    list of (start, stop) tuples, one per segment run, in data order.
    """
    if subject_ids is None:
        raise MetadataError("subject_ids is required")
    subject_ids = np.asarray(subject_ids)
    if subject_ids.ndim != 1:
        raise MetadataError(f"subject_ids must be 1-D, got shape {subject_ids.shape}")
    n = len(subject_ids)
    if n == 0:
        raise MetadataError("no windows")
    subject_ids = _as_1d("subject_ids", subject_ids, n)
    segment_ids = _as_1d("segment_ids", segment_ids, n)
    window_index = _as_1d("window_index", window_index, n)
    if window_index.dtype.kind not in "iu":
        raise MetadataError(f"window_index must be integer, got dtype {window_index.dtype}")

    new_subject = np.r_[True, subject_ids[1:] != subject_ids[:-1]]
    new_segment = new_subject | np.r_[True, segment_ids[1:] != segment_ids[:-1]]
    subj_starts = np.flatnonzero(new_subject)
    if len(set(subject_ids[subj_starts].tolist())) != len(subj_starts):
        raise MetadataError("each subject must occupy a single contiguous run of windows")
    seg_starts = np.flatnonzero(new_segment)
    keys = list(zip(subject_ids[seg_starts].tolist(), segment_ids[seg_starts].tolist()))
    if len(set(keys)) != len(keys):
        raise MetadataError("each (subject, segment) must occupy a single contiguous run of windows")
    bounds = np.r_[seg_starts, n]
    runs = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
    for a, b in runs:
        if b - a > 1 and not np.all(window_index[a + 1:b] > window_index[a:b - 1]):   # no subtraction: safe for unsigned dtypes
            raise MetadataError(f"window_index not strictly increasing inside segment {keys[runs.index((a, b))]}")
    return runs


def check_covariances(covs):
    covs = np.asarray(covs)
    if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
        raise ValueError(f"covariances must have shape (n_windows, n_channels, n_channels), got {covs.shape}")
    if not np.isfinite(covs).all():
        raise ValueError("covariances contain non-finite values")
    return covs


PARTITION_VALUES = ("calibration", "test")


def check_partition(subject_ids, segment_ids, window_index, partition, targets=None):
    """Validate a calibration/test partition chosen by the user and return it as an array of str.

    ``partition`` holds, per window, exactly ``"calibration"`` or ``"test"``. It must be constant inside every
    (subject, segment), so that no recorded segment is split between calibration and test, and every subject in
    ``targets`` (default: all) must have at least one calibration segment and one test segment. Segments come from
    ``segment_ids`` (acquisition metadata), never from labels or window identifiers.
    """
    runs = segment_runs(subject_ids, segment_ids, window_index)
    subject_ids, segment_ids = np.asarray(subject_ids), np.asarray(segment_ids)
    n = len(subject_ids)
    partition = _as_1d("partition", partition, n)
    if partition.dtype.kind not in "UO" or not all(isinstance(v, str) for v in partition.tolist()):
        raise MetadataError("partition must contain strings")
    partition = partition.astype(str)
    unknown = sorted(set(partition.tolist()) - set(PARTITION_VALUES))
    if unknown:
        raise MetadataError(f"partition values must be {PARTITION_VALUES}, got {unknown}")
    for a, b in runs:
        if len(set(partition[a:b].tolist())) != 1:
            key = (subject_ids[a:a + 1].tolist()[0], segment_ids[a:a + 1].tolist()[0])
            raise MetadataError(f"segment {key} is split between calibration and test")
    subjects = list(dict.fromkeys(subject_ids.tolist())) if targets is None else list(targets)
    for s in subjects:
        is_s = subject_ids == s
        if not is_s.any():
            raise MetadataError(f"subject {s!r} has no windows")
        for value in PARTITION_VALUES:
            if not (partition[is_s] == value).any():
                raise MetadataError(f"subject {s!r} has no {value} segment")
    return partition
