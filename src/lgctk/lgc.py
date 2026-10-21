"""Local Geometric Consistency (LGC): symmetric and random neighbourhoods.

Symmetric LGC replaces each covariance matrix C_i by the Riemannian (affine-invariant) mean of the matrices of windows
i-K, ..., i+K that belong to the same recorded segment as window i (clipped at the segment edges). Random LGC uses
neighbourhoods of the same size drawn at random (seeded) inside the same segment. Both use only covariances and
acquisition metadata: they have no labels argument and no fitted state. To switch LGC off, do not call them.

Both neighbourhoods may include future windows, so results obtained with them are offline.
"""
import numpy as np
from pyriemann.utils.mean import mean_riemann

from .validation import MetadataError, check_covariances, segment_runs


def _check_half_window(half_window):
    if isinstance(half_window, bool) or not isinstance(half_window, (int, np.integer)) or half_window < 0:
        raise ValueError(f"half_window must be an integer >= 0, got {half_window!r}")
    return int(half_window)


def symmetric_neighbourhoods(subject_ids, segment_ids, window_index, half_window=10):
    """Indices (into the full window array) of the neighbourhood of every window.

    Returns a list with one sorted int array per window; each array contains the window itself and never crosses a
    (subject, segment) boundary.
    """
    k = _check_half_window(half_window)
    out = []
    for start, stop in segment_runs(subject_ids, segment_ids, window_index):
        for i in range(start, stop):
            out.append(np.arange(max(start, i - k), min(stop, i + k + 1)))
    return out


def lgc_symmetric(covs, subject_ids, segment_ids, window_index, half_window=10):
    """Apply symmetric LGC.

    Parameters
    ----------
    covs : ndarray, shape (n_windows, n_channels, n_channels)
        SPD covariance matrices, in temporal order within each segment.
    subject_ids, segment_ids, window_index : array-like, shape (n_windows,)
        Mandatory acquisition metadata (see :func:`lgctk.validation.segment_runs`). ``segment_ids`` must come from
        recorded block/trial identifiers, never from labels.
    half_window : int, default=10
        Neighbourhood radius K in windows. ``0`` returns an unchanged copy.

    Integer or boolean covariances are promoted to float64 when ``half_window > 0`` (a Riemannian mean is not an
    integer); float32 and float64 inputs keep their dtype.

    Returns
    -------
    ndarray, same shape and order as ``covs``.
    """
    covs = check_covariances(covs)
    neighbourhoods = symmetric_neighbourhoods(subject_ids, segment_ids, window_index, half_window)
    if len(neighbourhoods) != len(covs):
        raise MetadataError("metadata length differs from the number of covariance matrices")
    if half_window == 0:
        return covs.copy()
    if covs.dtype.kind in "biu":
        covs = covs.astype(np.float64)
    out = np.empty_like(covs)
    for i, idx in enumerate(neighbourhoods):
        out[i] = mean_riemann(covs[idx])
    return out


def _check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
    return int(seed)


def random_neighbourhoods(subject_ids, segment_ids, window_index, half_window=10, *, seed):
    """Random neighbourhoods with the size of the symmetric ones.

    For every window, the neighbourhood has the size the symmetric neighbourhood would have at that position; the other
    members are drawn without replacement from the rest of the same (subject, segment), and the window itself is always
    included. A fresh ``np.random.default_rng(seed)`` is created per subject and per call and visits that subject's
    windows in data order; when no extra member is needed the generator is not consumed. The draws therefore depend on
    which windows enter the call: calling this on a calibration subset and on a test subset separately is not the same
    as calling it on the whole subject (see docs/REPRODUCIBILITY.md).

    Returns a list with one sorted int array per window (indices into the full window array).
    """
    k = _check_half_window(half_window)
    seed = _check_seed(seed)
    runs = segment_runs(subject_ids, segment_ids, window_index)
    subject_ids = np.asarray(subject_ids)
    out, rng, current = [], None, None
    for start, stop in runs:
        if rng is None or subject_ids[start] != current:
            current, rng = subject_ids[start], np.random.default_rng(seed)
        segment = np.arange(start, stop)
        for i in range(start, stop):
            n_i = min(stop, i + k + 1) - max(start, i - k)
            pool = segment[segment != i]
            take = min(n_i - 1, len(pool))
            chosen = rng.choice(pool, size=take, replace=False) if take > 0 else np.empty(0, dtype=int)
            out.append(np.sort(np.concatenate([[i], chosen]).astype(int)))
    return out


def lgc_random(covs, subject_ids, segment_ids, window_index, half_window=10, *, seed):
    """Apply LGC with random neighbourhoods (see :func:`random_neighbourhoods`).

    Each covariance matrix is replaced by the Riemannian (affine-invariant) mean of its random neighbourhood, drawn
    inside its own recorded segment. Same inputs, dtype rules and guarantees as :func:`lgc_symmetric` (no labels
    argument, no fitted state, never reorders data); ``seed`` is mandatory.
    """
    covs = check_covariances(covs)
    neighbourhoods = random_neighbourhoods(subject_ids, segment_ids, window_index, half_window, seed=seed)
    if len(neighbourhoods) != len(covs):
        raise MetadataError("metadata length differs from the number of covariance matrices")
    if half_window == 0:
        return covs.copy()
    if covs.dtype.kind in "biu":
        covs = covs.astype(np.float64)
    out = np.empty_like(covs)
    for i, idx in enumerate(neighbourhoods):
        out[i] = mean_riemann(covs[idx])
    return out
