"""Leave-one-subject-out evaluation with a label firewall.

The evaluator is the only component that holds evaluation labels and scores the predictions itself.

- :func:`loso_unlabelled_target`: for each target it hands the pipeline the covariances and domains of ALL subjects
  plus the labels of the SOURCE windows.
- :func:`loso_calibrated`: for each target it hands the pipeline the labelled sources, the labelled CALIBRATION windows
  of the target and the target TEST covariances; the test labels stay here.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from .validation import check_covariances, check_partition

FIT_ORDERS = ("sources_then_target", "as_given")


@dataclass
class LOSOResult:
    """Per-window predictions (aligned with ``window_ids``) and per-subject scores."""

    window_ids: np.ndarray
    subject_ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    per_subject: list = field(default_factory=list)

    @property
    def mean_accuracy(self):
        return float(np.mean([r["accuracy"] for r in self.per_subject]))


@dataclass
class CalibratedLOSOResult(LOSOResult):
    """:class:`LOSOResult` of the test windows plus, per target, the window IDs used as calibration."""

    calibration_window_ids: dict = field(default_factory=dict)


def score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    out = dict(n=len(y_true), hits=int((y_true == y_pred).sum()), accuracy=float(np.mean(y_true == y_pred)))
    if len(np.unique(y_true)) > 1:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    else:
        out["balanced_accuracy"] = out["f1_macro"] = float("nan")
    return out


def loso_unlabelled_target(covs, subject_ids, labels, pipeline, *, window_ids, targets=None,
                           fit_order="sources_then_target"):
    """Leave-one-subject-out evaluation, target unlabelled.

    Parameters
    ----------
    covs : ndarray, shape (n_windows, n_channels, n_channels)
        Window representations (e.g. after :func:`lgctk.lgc.lgc_symmetric`), subjects in contiguous runs.
    subject_ids : array-like of str, shape (n_windows,)
        Used as domain names.
    labels : array-like of int, shape (n_windows,)
        Evaluation labels; only source labels are passed to ``pipeline``.
    pipeline : object with ``fit_predict(X, domains, target_domain, y_source)``
    window_ids : array-like, shape (n_windows,)
        Unique window identifiers, stored with the predictions (parity is checked by ID, never by row position).
    targets : iterable of str, optional
        Target subjects (default: every subject, in data order).
    fit_order : {"sources_then_target", "as_given"}
        Order in which windows enter each fit: the source subjects in data order followed by the target, or the data
        order itself. Both are valid; they are kept because the reference producers used different orders and exact
        floating-point parity depends on it.
    """
    if fit_order not in FIT_ORDERS:
        raise ValueError(f"fit_order must be one of {FIT_ORDERS}")
    covs = np.asarray(covs)
    subject_ids = np.asarray(subject_ids).astype(str)
    labels = np.asarray(labels).astype(int)
    window_ids = np.asarray(window_ids)
    n = len(covs)
    if not (len(subject_ids) == len(labels) == len(window_ids) == n):
        raise ValueError("covs, subject_ids, labels and window_ids differ in length")
    if len(np.unique(window_ids)) != n:
        raise ValueError("window_ids must be unique")
    order_subjects = list(dict.fromkeys(subject_ids.tolist()))
    targets = order_subjects if targets is None else [str(t) for t in targets]

    rows_ids, rows_subj, rows_true, rows_pred, per_subject = [], [], [], [], []
    for t in targets:
        is_t = subject_ids == t
        if not is_t.any():
            raise ValueError(f"target {t!r} has no windows")
        if fit_order == "sources_then_target":
            order = np.r_[np.flatnonzero(~is_t), np.flatnonzero(is_t)]
        else:
            order = np.arange(n)
        src_in_order = ~is_t[order]
        y_source = labels[order][src_in_order]                # source labels only (label firewall)
        y_hat = np.asarray(pipeline.fit_predict(covs[order], subject_ids[order], t, y_source)).astype(int)
        tgt_rows = order[~src_in_order]                       # target rows, in the order the pipeline saw them
        if len(y_hat) != len(tgt_rows):
            raise RuntimeError(f"pipeline returned {len(y_hat)} predictions for {len(tgt_rows)} target windows")
        y_true = labels[tgt_rows]                             # evaluator only
        per_subject.append(dict(target=t, **score(y_true, y_hat)))
        rows_ids.append(window_ids[tgt_rows])
        rows_subj.append(subject_ids[tgt_rows])
        rows_true.append(y_true)
        rows_pred.append(y_hat)
    return LOSOResult(window_ids=np.concatenate(rows_ids), subject_ids=np.concatenate(rows_subj),
                      y_true=np.concatenate(rows_true), y_pred=np.concatenate(rows_pred), per_subject=per_subject)


def loso_calibrated(covs, subject_ids, labels, partition, pipeline, *, segment_ids, window_index, window_ids,
                    target_covs=None, targets=None):
    """Leave-one-subject-out evaluation with a labelled calibration part of each target.

    For each target subject: all windows of the other subjects are sources (with labels, in data order); the target's
    windows marked ``"calibration"`` enter the fit with their labels; the windows marked ``"test"`` are predicted
    without labels and scored here.

    Parameters
    ----------
    covs : ndarray, shape (n_windows, n_channels, n_channels)
        Representation of every window as a SOURCE (e.g. after LGC over each whole subject).
    subject_ids : array-like of str, shape (n_windows,)
    labels : array-like of int, shape (n_windows,)
        Only source and calibration labels are passed to ``pipeline``; test labels are used for scoring only.
    partition : array-like of str, shape (n_windows,)
        ``"calibration"`` or ``"test"`` per window, used when that window's subject is the target. Chosen by the user
        (benchmark design), constant inside every recorded segment; validated by
        :func:`lgctk.validation.check_partition` before any fit.
    pipeline : object with ``fit_predict(X_source, y_source, source_domains, X_calib, y_calib, X_test, target_domain)``
    segment_ids, window_index : array-like, shape (n_windows,)
        Acquisition metadata (see :func:`lgctk.validation.segment_runs`).
    window_ids : array-like, shape (n_windows,)
        Window identifiers, unique within each subject; stored with the predictions.
    target_covs : ndarray, optional, same shape as ``covs``
        Representation of every window when its subject is the TARGET (default: ``covs``). Needed when the target's
        calibration and test parts are transformed separately, e.g. random LGC applied per partition. Rows are selected
        with the same masks as ``covs``: the caller is responsible for building it in the same window order.
    targets : iterable of str, optional
        Target subjects (default: every subject, in data order).
    """
    covs = check_covariances(covs)
    subject_ids = np.asarray(subject_ids).astype(str)
    labels = np.asarray(labels).astype(int)
    window_ids = np.asarray(window_ids)
    n = len(covs)
    if not (len(subject_ids) == len(labels) == len(window_ids) == n):
        raise ValueError("covs, subject_ids, labels and window_ids differ in length")
    keys = list(zip(subject_ids.tolist(), window_ids.tolist()))
    if len(set(keys)) != n:
        raise ValueError("window_ids must be unique within each subject")
    order_subjects = list(dict.fromkeys(subject_ids.tolist()))
    targets = order_subjects if targets is None else [str(t) for t in targets]
    part = check_partition(subject_ids, segment_ids, window_index, partition, targets=targets)   # before any fit
    if target_covs is None:
        target_covs = covs
    else:
        target_covs = check_covariances(target_covs)
        if target_covs.shape != covs.shape:
            raise ValueError(f"target_covs has shape {target_covs.shape}, expected {covs.shape}")

    rows_ids, rows_subj, rows_true, rows_pred, per_subject, cal_ids = [], [], [], [], [], {}
    for t in targets:
        is_t = subject_ids == t
        src = ~is_t
        cal = is_t & (part == "calibration")
        test = is_t & (part == "test")
        if set(window_ids[cal].tolist()) & set(window_ids[test].tolist()):
            raise RuntimeError(f"target {t!r}: calibration and test window IDs overlap")
        y_hat = np.asarray(pipeline.fit_predict(covs[src], labels[src], subject_ids[src], target_covs[cal],
                                                labels[cal], target_covs[test], t)).astype(int)
        if len(y_hat) != int(test.sum()):
            raise RuntimeError(f"pipeline returned {len(y_hat)} predictions for {int(test.sum())} test windows")
        y_true = labels[test]                                 # evaluator only
        per_subject.append(dict(target=t, n_calibration=int(cal.sum()), **score(y_true, y_hat)))
        cal_ids[t] = window_ids[cal]
        rows_ids.append(window_ids[test])
        rows_subj.append(subject_ids[test])
        rows_true.append(y_true)
        rows_pred.append(y_hat)
    return CalibratedLOSOResult(window_ids=np.concatenate(rows_ids), subject_ids=np.concatenate(rows_subj),
                                y_true=np.concatenate(rows_true), y_pred=np.concatenate(rows_pred),
                                per_subject=per_subject, calibration_window_ids=cal_ids)
