"""Helpers for the evaluation-label invariance test (see docs/REPRODUCIBILITY.md#tests).

Fake labels are injected at the data source (the class column / per-segment label vector the pipeline reads), in
copies; the run is repeated from scratch and predictions, representations and fitted state are compared exactly.
"""
import numpy as np

BINARY_SCHEMES = ("invert", "constant_0", "constant_1", "random_per_segment", "random_per_window")


class InvalidTest(Exception):
    """A guard of the invariance test failed (the case is ERROR/INVALID_TEST, never a pass)."""


def fake_labels(y_true, scheme, segment_ids=None, seed=0):
    """Binary fake labels for one subject's windows.

    ``random_per_segment`` draws one label per (contiguous) segment; ``random_per_window`` one per window. The RNG is
    private to this call. Raises :class:`InvalidTest` if the substitution changes no value.
    """
    y = np.asarray(y_true).astype(int)
    if not np.isin(y, [0, 1]).all():
        raise InvalidTest("binary schemes need labels in {0, 1}")
    rng = np.random.default_rng(seed)
    if scheme == "invert":
        f = 1 - y
    elif scheme == "constant_0":
        f = np.zeros_like(y)
    elif scheme == "constant_1":
        f = np.ones_like(y)
    elif scheme == "random_per_segment":
        if segment_ids is None:
            raise InvalidTest("random_per_segment needs segment_ids")
        seg = np.asarray(segment_ids)
        starts = np.r_[True, seg[1:] != seg[:-1]]
        f = rng.integers(0, 2, size=int(starts.sum()))[np.cumsum(starts) - 1]
    elif scheme == "random_per_window":
        f = rng.integers(0, 2, size=len(y))
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    if np.array_equal(f, y):
        raise InvalidTest(f"scheme {scheme!r} changed no label")
    return f.astype(int)


class FitSpy:
    """Wrap an estimator instance's ``fit`` to record the labels and sample weights it actually receives."""

    def __init__(self, estimator):
        self.calls = []
        original = estimator.fit

        def spy_fit(X, y, *args, **kwargs):
            w = kwargs.get("sample_weight")
            self.calls.append(dict(y=np.array(y, copy=True), sample_weight=None if w is None else np.array(w)))
            return original(X, y, *args, **kwargs)

        estimator.fit = spy_fit


def arrays_equal(a, b):
    """Exact equality of two arrays, including shape."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


class StageSpy:
    """Record the rows every fitted stage of a scikit-learn pipeline actually receives.

    Wraps ``fit`` and ``fit_transform`` of each step INSTANCE (scikit-learn does not clone steps when caching is off)
    and of optional ``extra`` objects such as the classifier inside ``TLClassifier``. Each call records the input X,
    y and sample weights and, for ``fit_transform``, the output, so that the correspondence of rows can be followed
    explicitly from one stage to the next. Nested calls (``fit_transform`` calling ``fit``) are recorded once.
    """

    def __init__(self, pipeline, extra=()):
        self.calls = []
        self._depth = 0
        for name, step in list(pipeline.steps) + list(extra):
            for method in ("fit", "fit_transform"):
                if hasattr(step, method):
                    setattr(step, method, self._wrap(name, method, getattr(step, method)))

    def _wrap(self, name, method, original):
        def spy(X, y=None, *args, **kwargs):
            self._depth += 1
            try:
                out = original(X, y, *args, **kwargs)
            finally:
                self._depth -= 1
            if self._depth == 0 or name.startswith("extra:"):
                w = kwargs.get("sample_weight")
                self.calls.append(dict(stage=name, method=method, X=np.array(X, copy=True),
                                       y=None if y is None else np.array(y, copy=True),
                                       sample_weight=None if w is None else np.array(w, copy=True),
                                       out=np.array(out, copy=True) if method == "fit_transform" else None))
            return out
        return spy


def row_hashes(X):
    """One hash per row (sample) of an array, for row-membership checks."""
    X = np.ascontiguousarray(X)
    return [hash(X[i].tobytes()) for i in range(len(X))]


def fit_access_check(calls, X_fit_expected, X_forbidden, n_pipeline_stages):
    """Data-access check of one calibrated fit from the rows recorded by :class:`StageSpy`.

    ``X_fit_expected`` (sources + calibration, in fit order) and ``X_forbidden`` (the target test windows) are built
    by the caller from its own partition, never taken from the pipeline. Returns a dict of booleans:

    - ``stages_observed``: every pipeline stage was fitted exactly once;
    - ``first_stage_rows``: the first stage received exactly ``X_fit_expected``;
    - ``rows_propagated``: each stage received the output of the previous ``fit_transform``;
    - ``no_forbidden_rows``: no forbidden row reached the first stage;
    - ``same_row_count``: every recorded fit (classifier included) received ``len(X_fit_expected)`` rows.
    """
    stages = [c for c in calls if not c["stage"].startswith("extra:")]
    first = stages[0]["X"] if stages else np.empty(0)
    forbidden = set(row_hashes(X_forbidden))
    return dict(
        stages_observed=len(stages) == n_pipeline_stages,
        first_stage_rows=arrays_equal(first, X_fit_expected),
        rows_propagated=all(arrays_equal(stages[i]["X"], stages[i - 1]["out"]) for i in range(1, len(stages))),
        no_forbidden_rows=not (set(row_hashes(first)) & forbidden),
        same_row_count=all(len(c["X"]) == len(X_fit_expected) for c in calls),
    )
