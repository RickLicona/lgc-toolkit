"""Evaluation-label invariance test with positive controls (protocol v1.1, synthetic data).

For every target subject and fake-label scheme the whole path is re-run from the recordings: arms A and A2 (true
labels) and B and B2 (fake labels injected at the source). The package path must give identical predictions,
LGC representations, segments, fitted state and fit inputs. Positive controls -- the original FgMDM construction
(L1), segments inferred from labels (L2) and, in the calibrated path, test windows with their labels inside the fit --
must be detected through the same injection boundary, otherwise a pass means nothing. A label test cannot see test
covariances entering an unsupervised fit without labels; a separate data-access check covers that case and is itself
validated by a negative fixture.
"""
import unittest
from functools import partial
from itertools import product

import numpy as np

from lgctk.leakage import BINARY_SCHEMES, InvalidTest, arrays_equal, fake_labels

from synthetic import (CAL_BLOCKS, LeakyUnsupervisedTSALR, block_labels, featurize, make_recordings,
                       run_calibrated_path, run_original_fgmdm_path, run_package_path, with_target_block_labels)

SOURCE_SCHEMES = ("invert", "constant_0", "constant_1", "random_per_segment")
WINDOW_SCHEMES = ("random_per_window",)


def first_effective(y, scheme, max_seed=100):
    """Pre-specified rule for random schemes: the first seed whose substitution changes at least one value."""
    for seed in range(max_seed):
        try:
            return fake_labels(y, "random_per_window" if scheme == "random_per_segment" else scheme, seed=seed)
        except InvalidTest:
            if scheme not in ("random_per_segment", "random_per_window"):
                raise
    raise InvalidTest(f"no effective seed for {scheme}")


def arms(recs, target, scheme, path):
    """Return A, A2, B, B2 and the fake labels expected by the evaluator for the target windows."""
    if scheme in SOURCE_SCHEMES:
        blocks, y_blocks = block_labels(recs[target])
        fake_blocks = first_effective(y_blocks, scheme)
        recs_b = with_target_block_labels(recs, target, fake_blocks)
        b_kwargs = {}
        _, subj, seg, _, _, _ = featurize(recs_b)
        expected = fake_blocks[np.searchsorted(blocks, seg[subj == target])]
    else:
        y_true = featurize(recs)[5][featurize(recs)[1] == target]
        expected = first_effective(y_true, scheme)
        recs_b, b_kwargs = recs, dict(window_labels=(target, expected))
    A, A2 = path(recs, target), path(recs, target)
    B, B2 = path(recs_b, target, **b_kwargs), path(recs_b, target, **b_kwargs)
    return A, A2, B, B2, expected


def same(a, b, keys):
    return all(arrays_equal(a[k], b[k]) for k in keys) and \
        set(a["state"]) == set(b["state"]) and len(a["state"]) > 0 and \
        all(arrays_equal(a["state"][k], b["state"][k]) for k in a["state"])


# Package configurations exercised by the synthetic invariance test: the v0.1 shapes (RCT without LGC, symmetric LGC
# K=10) under both fit orders used by the reproduction scripts, plus K=3 (short neighbourhood, segment edges active).
CONFIGS = {
    "LGC_K3/sources_then_target": dict(half_window=3, fit_order="sources_then_target"),
    "RCT/sources_then_target": dict(half_window=0, fit_order="sources_then_target"),
    "LGC_K10/sources_then_target": dict(half_window=10, fit_order="sources_then_target"),
    "RCT/as_given": dict(half_window=0, fit_order="as_given"),
    "LGC_K10/as_given": dict(half_window=10, fit_order="as_given"),
    "LGC_random_K3/sources_then_target": dict(half_window=3, lgc="random", fit_order="sources_then_target"),
    "LGC_random_K10/as_given": dict(half_window=10, lgc="random", fit_order="as_given"),
}


class TestPackageInvariance(unittest.TestCase):
    """Package path: PASS_PREDICTIONS expected for every configuration, target and scheme. Synthetic data: this is a
    test of the package mechanism, not an invariance run on each real configuration."""

    @classmethod
    def setUpClass(cls):
        cls.recs = make_recordings()

    def test_all_configs_targets_schemes(self):
        self.assertEqual(set(SOURCE_SCHEMES + WINDOW_SCHEMES), set(BINARY_SCHEMES))
        keys = ("y_hat", "window_ids", "X_lgc", "segments", "y_fed", "w_fed")
        for name, target, scheme in product(CONFIGS, sorted(self.recs), SOURCE_SCHEMES + WINDOW_SCHEMES):
            with self.subTest(config=name, target=target, scheme=scheme):
                path = partial(run_package_path, **CONFIGS[name])
                A, A2, B, B2, expected = arms(self.recs, target, scheme, path)
                # guards: repeatability, effective substitution observed by the evaluator
                self.assertTrue(same(A, A2, keys) and same(B, B2, keys), "A/A2 or B/B2 not repeatable")
                self.assertTrue(arrays_equal(B["y_true_seen_by_evaluator"], expected))
                self.assertFalse(arrays_equal(A["y_true_seen_by_evaluator"], expected))
                # invariance: predictions, representations, segments, fitted state and fit inputs
                self.assertTrue(same(A, B, keys), "LABEL_DEPENDENCE")


class TestPositiveControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = make_recordings()

    def test_c1_fgmdm_with_zero_target_weight_is_detected(self):
        """C+1 (L1): a reproducible change of predictions or of the captured FgMDM state, for every target."""
        for target in sorted(self.recs):
            with self.subTest(target=target):
                A, A2, B, B2, _ = arms(self.recs, target, "invert", run_original_fgmdm_path)
                self.assertTrue(same(A, A2, ("y_hat",)) and same(B, B2, ("y_hat",)), "not repeatable")
                changed = not arrays_equal(A["y_hat"], B["y_hat"]) or not same(A, B, ())
                self.assertTrue(changed, "CONTROL_NOT_DETECTED")

    def test_c2_segments_inferred_from_labels_are_detected(self):
        """C+2 (L2): inferred boundaries AND post-LGC covariances change (schemes that alter the label structure)."""
        path = lambda r, t, **kw: run_package_path(r, t, segments="from_labels", **kw)  # noqa: E731
        for target in sorted(self.recs):
            for scheme in ("constant_0", "random_per_window"):
                with self.subTest(target=target, scheme=scheme):
                    A, A2, B, B2, _ = arms(self.recs, target, scheme, path)
                    self.assertTrue(same(A, A2, ("segments", "X_lgc")) and same(B, B2, ("segments", "X_lgc")))
                    self.assertFalse(arrays_equal(A["segments"], B["segments"]), "CONTROL_NOT_DETECTED")
                    self.assertFalse(arrays_equal(A["X_lgc"], B["X_lgc"]), "CONTROL_NOT_DETECTED")

    def test_c2_with_random_lgc_is_detected(self):
        """C+2 through the random LGC path: segments inferred from labels change the random neighbourhoods."""
        path = lambda r, t, **kw: run_package_path(r, t, segments="from_labels", lgc="random", **kw)  # noqa: E731
        for target in sorted(self.recs):
            with self.subTest(target=target):
                A, A2, B, B2, _ = arms(self.recs, target, "constant_0", path)
                self.assertTrue(same(A, A2, ("segments", "X_lgc")) and same(B, B2, ("segments", "X_lgc")))
                self.assertFalse(arrays_equal(A["segments"], B["segments"]), "CONTROL_NOT_DETECTED")
                self.assertFalse(arrays_equal(A["X_lgc"], B["X_lgc"]), "CONTROL_NOT_DETECTED")

    def test_c2_structure_preserving_scheme_does_not_detect(self):
        """Why several schemes are needed: inverting alternating block labels keeps the inferred boundaries."""
        path = lambda r, t, **kw: run_package_path(r, t, segments="from_labels", **kw)  # noqa: E731
        A, _, B, _, _ = arms(self.recs, "subject_01", "invert", path)
        self.assertTrue(arrays_equal(A["segments"], B["segments"]))


# ---- calibrated path (TSA-LR): fake labels only on the target TEST windows ------------------------------------------
CALIBRATED_CONFIGS = {
    "TSA-LR": dict(lgc="none"),
    "LGC_K3+TSA-LR": dict(lgc="symmetric", half_window=3),
    "LGC_random_K3+TSA-LR": dict(lgc="random", half_window=3),
}
CALIBRATED_KEYS = ("y_hat", "window_ids", "X_src", "X_tgt", "partition", "y_fed", "fit_rows")


def calibrated_arms(recs, target, scheme, path):
    """A, A2, B, B2 for the calibrated path. The partition comes from metadata and is frozen; calibration and source
    labels are legitimate inputs and are never changed."""
    blocks, y_blocks = block_labels(recs[target])
    is_test_block = ~np.isin(blocks, CAL_BLOCKS)
    if scheme in SOURCE_SCHEMES:
        fake = y_blocks.copy()
        fake[is_test_block] = first_effective(y_blocks[is_test_block], scheme)
        recs_b, b_kwargs = with_target_block_labels(recs, target, fake), {}
        _, subj, seg, _, _, _ = featurize(recs_b)
        m = (subj == target) & ~np.isin(seg, CAL_BLOCKS)
        expected = fake[np.searchsorted(blocks, seg[m])]
    else:
        _, subj, seg, _, _, y = featurize(recs)
        y_t, seg_t = y[subj == target].copy(), seg[subj == target]
        is_test = ~np.isin(seg_t, CAL_BLOCKS)
        y_t[is_test] = first_effective(y_t[is_test], scheme)
        expected, recs_b, b_kwargs = y_t[is_test], recs, dict(window_labels=(target, y_t))
    A, A2 = path(recs, target), path(recs, target)
    B, B2 = path(recs_b, target, **b_kwargs), path(recs_b, target, **b_kwargs)
    return A, A2, B, B2, expected


class TestCalibratedInvariance(unittest.TestCase):
    """TSA-LR calibrated path: PASS_PREDICTIONS and a clean data-access check for every configuration, target and
    scheme (synthetic data; a test of the package mechanism)."""

    @classmethod
    def setUpClass(cls):
        cls.recs = make_recordings()

    def test_all_configs_targets_schemes(self):
        for name, target, scheme in product(CALIBRATED_CONFIGS, sorted(self.recs), SOURCE_SCHEMES + WINDOW_SCHEMES):
            with self.subTest(config=name, target=target, scheme=scheme):
                path = partial(run_calibrated_path, **CALIBRATED_CONFIGS[name])
                A, A2, B, B2, expected = calibrated_arms(self.recs, target, scheme, path)
                self.assertTrue(same(A, A2, CALIBRATED_KEYS) and same(B, B2, CALIBRATED_KEYS), "not repeatable")
                self.assertTrue(arrays_equal(B["y_true_seen_by_evaluator"], expected))
                self.assertFalse(arrays_equal(A["y_true_seen_by_evaluator"], expected))
                for arm in (A, B):
                    self.assertTrue(all(arm["access"].values()), f"ACCESS_VIOLATION {arm['access']}")
                self.assertTrue(same(A, B, CALIBRATED_KEYS), "LABEL_DEPENDENCE")


class TestCalibratedControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = make_recordings()

    def test_cc1_labelled_test_windows_in_the_fit_are_detected(self):
        """Positive control: test windows WITH labels inside the fit -> access violation AND label dependence."""
        for name, target in product(CALIBRATED_CONFIGS, sorted(self.recs)):
            with self.subTest(config=name, target=target):
                path = partial(run_calibrated_path, evaluator="leaky_labelled", **CALIBRATED_CONFIGS[name])
                A, A2, B, B2, _ = calibrated_arms(self.recs, target, "invert", path)
                self.assertTrue(same(A, A2, ("y_hat",)) and same(B, B2, ("y_hat",)), "not repeatable")
                self.assertFalse(A["access"]["no_forbidden_rows"], "CONTROL_NOT_DETECTED (access)")
                self.assertFalse(A["access"]["first_stage_rows"], "CONTROL_NOT_DETECTED (access)")
                changed = not arrays_equal(A["y_hat"], B["y_hat"]) or not same(A, B, ())
                self.assertTrue(changed, "CONTROL_NOT_DETECTED (labels)")

    def test_cc2_test_covariances_in_an_unsupervised_fit_are_detected(self):
        """Negative fixture: test covariances WITHOUT labels in the unsupervised stages. The label test cannot see it
        (no label is used); the data-access check must fail regardless of the predictions."""
        for name, target in product(CALIBRATED_CONFIGS, sorted(self.recs)):
            with self.subTest(config=name, target=target):
                path = partial(run_calibrated_path, pipeline_cls=LeakyUnsupervisedTSALR, **CALIBRATED_CONFIGS[name])
                A, A2, B, B2, _ = calibrated_arms(self.recs, target, "invert", path)
                self.assertFalse(A["access"]["no_forbidden_rows"], "ACCESS_VIOLATION_NOT_DETECTED")
                self.assertFalse(A["access"]["same_row_count"], "ACCESS_VIOLATION_NOT_DETECTED")
                self.assertTrue(same(A, B, ("y_hat",)), "fixture expected to be label-invariant")


if __name__ == "__main__":
    unittest.main()
