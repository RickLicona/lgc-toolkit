"""Label firewall by construction: what each component receives."""
import inspect
import unittest

import numpy as np

from lgctk import RCTMDM, TSALR, MetadataError, loso_calibrated, loso_unlabelled_target

from synthetic import K, ObservedRCTMDM, ObservedTSALR, calibration_partition, featurize, make_recordings
from lgctk import lgc_symmetric


class RecordingPipeline:
    """Records every argument the evaluator hands to the predictive pipeline."""

    def __init__(self):
        self.calls = []

    def fit_predict(self, X, domains, target_domain, y_source):
        self.calls.append(dict(X=np.array(X), domains=np.array(domains), target=target_domain,
                               y_source=np.array(y_source)))
        return np.zeros(int((np.asarray(domains) == target_domain).sum()), dtype=int)


class TestFirewall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        covs, subj, seg, widx, wid, y = featurize(make_recordings())
        cls.X = lgc_symmetric(covs, subj, seg, widx, half_window=K)
        cls.subj, cls.wid, cls.y = subj, wid, y

    def test_pipeline_signature_has_no_target_labels(self):
        params = list(inspect.signature(RCTMDM.fit_predict).parameters)
        self.assertEqual(params, ["self", "X", "domains", "target_domain", "y_source"])

    def test_evaluator_passes_only_source_labels(self):
        for fit_order in ("sources_then_target", "as_given"):
            runs = []
            for flip in (False, True):
                y = self.y.copy()
                t = "subject_02"
                if flip:
                    y[self.subj == t] = 1 - y[self.subj == t]
                pipe = RecordingPipeline()
                loso_unlabelled_target(self.X, self.subj, y, pipe, window_ids=self.wid, targets=[t],
                                       fit_order=fit_order)
                call = pipe.calls[0]
                src = call["domains"] != t
                self.assertEqual(len(call["y_source"]), int(src.sum()))
                runs.append(call)
            a, b = runs                    # everything the pipeline received is identical when target labels change
            for k in ("X", "domains", "y_source"):
                self.assertTrue(np.array_equal(a[k], b[k]), k)

    def test_fit_order(self):
        pipe = RecordingPipeline()
        loso_unlabelled_target(self.X, self.subj, self.y, pipe, window_ids=self.wid, targets=["subject_02"])
        d = pipe.calls[0]["domains"]
        self.assertTrue(np.all(d[-int((d == "subject_02").sum()):] == "subject_02"))
        pipe = RecordingPipeline()
        loso_unlabelled_target(self.X, self.subj, self.y, pipe, window_ids=self.wid, targets=["subject_02"],
                               fit_order="as_given")
        self.assertTrue(np.array_equal(pipe.calls[0]["domains"], self.subj))

    def test_mdm_receives_placeholder_with_zero_weight(self):
        pipe = ObservedRCTMDM()
        t = "subject_03"
        loso_unlabelled_target(self.X, self.subj, self.y, pipe, window_ids=self.wid, targets=[t])
        log = pipe.log[0]
        is_t = np.r_[self.subj[self.subj != t], self.subj[self.subj == t]] == t
        y_fed, w_fed = np.asarray(log["y_fed"]).astype(int), np.asarray(log["w_fed"])
        self.assertTrue(np.all(w_fed[is_t] == 0) and np.all(w_fed[~is_t] == 1))
        self.assertEqual(len(np.unique(y_fed[is_t])), 1)            # constant placeholder, not the target labels
        self.assertTrue(np.array_equal(y_fed[~is_t], np.r_[self.y[self.subj != t]]))

    def test_guards(self):
        with self.assertRaises(ValueError):
            loso_unlabelled_target(self.X, self.subj, self.y, RecordingPipeline(),
                                   window_ids=np.zeros(len(self.y)))
        with self.assertRaises(ValueError):
            RCTMDM().fit_predict(self.X, np.char.add(self.subj, "/x"), "subject_01/x", self.y[self.subj != "subject_01"])
        with self.assertRaises(ValueError):
            RCTMDM().fit_predict(self.X, self.subj, "subject_01", self.y)


class RecordingCalibratedPipeline:
    """Records every argument the calibrated evaluator hands to the predictive pipeline."""

    def __init__(self):
        self.calls = []

    def fit_predict(self, X_source, y_source, source_domains, X_calib, y_calib, X_test, target_domain):
        self.calls.append(dict(X_source=np.array(X_source), y_source=np.array(y_source),
                               source_domains=np.array(source_domains), X_calib=np.array(X_calib),
                               y_calib=np.array(y_calib), X_test=np.array(X_test), target=target_domain))
        return np.zeros(len(X_test), dtype=int)


class TestCalibratedFirewall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        covs, subj, seg, widx, wid, y = featurize(make_recordings())
        cls.X = lgc_symmetric(covs, subj, seg, widx, half_window=K)
        cls.subj, cls.seg, cls.widx, cls.wid, cls.y = subj, seg, widx, wid, y
        cls.part = calibration_partition(subj, seg)

    def run_eval(self, pipe, y=None, **over):
        kw = dict(segment_ids=self.seg, window_index=self.widx, window_ids=self.wid, targets=["subject_02"])
        kw.update(over)
        return loso_calibrated(self.X, self.subj, self.y if y is None else y, kw.pop("partition", self.part), pipe,
                               **kw)

    def test_signatures_have_no_test_labels(self):
        params = list(inspect.signature(TSALR.fit_predict).parameters)
        self.assertEqual(params, ["self", "X_source", "y_source", "source_domains", "X_calib", "y_calib", "X_test",
                                  "target_domain"])

    def test_evaluator_passes_only_source_and_calibration_labels(self):
        t = "subject_02"
        test = (self.subj == t) & (self.part == "test")
        cal = (self.subj == t) & (self.part == "calibration")
        runs = []
        for flip in (False, True):
            y = self.y.copy()
            if flip:
                y[test] = 1 - y[test]
            pipe = RecordingCalibratedPipeline()
            res = self.run_eval(pipe, y=y)
            runs.append(pipe.calls[0])
            self.assertTrue(np.array_equal(res.y_true, y[test]))         # the evaluator scores with test labels
        a, b = runs                    # everything the pipeline received is identical when test labels change
        for k in ("X_source", "y_source", "source_domains", "X_calib", "y_calib", "X_test"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)
        self.assertTrue(np.array_equal(a["X_calib"], self.X[cal]) and np.array_equal(a["X_test"], self.X[test]))
        self.assertTrue(np.array_equal(a["y_source"], self.y[self.subj != t]))
        self.assertNotIn(t, set(a["source_domains"].tolist()))

    def test_target_covs_used_for_the_target_only(self):
        alt = self.X * 2.0
        pipe = RecordingCalibratedPipeline()
        self.run_eval(pipe, target_covs=alt)
        t = "subject_02"
        call = pipe.calls[0]
        self.assertTrue(np.array_equal(call["X_source"], self.X[self.subj != t]))
        self.assertTrue(np.array_equal(call["X_calib"], alt[(self.subj == t) & (self.part == "calibration")]))
        self.assertTrue(np.array_equal(call["X_test"], alt[(self.subj == t) & (self.part == "test")]))

    def test_guards_fail_before_any_fit(self):
        bad_part = self.part.copy()
        first_test = np.flatnonzero((self.subj == "subject_02") & (self.part == "test"))[0]
        bad_part[first_test - 1] = "test"          # last calibration window of block 2 -> block split
        no_test = np.where(self.subj == "subject_02", "calibration", self.part)
        dup = self.wid.copy()
        dup[1] = dup[0]
        cases = dict(split_segment=dict(partition=bad_part), empty_side=dict(partition=no_test),
                     duplicate_ids=dict(window_ids=dup), no_segments=dict(segment_ids=None),
                     no_window_index=dict(window_index=None), bad_target_covs=dict(target_covs=self.X[:-1]))
        for name, over in cases.items():
            with self.subTest(case=name):
                pipe = RecordingCalibratedPipeline()
                with self.assertRaises((MetadataError, ValueError)):
                    self.run_eval(pipe, **over)
                self.assertEqual(pipe.calls, [], "pipeline called before the guard")

    def test_tsalr_guards(self):
        t = "subject_02"
        src, cal = self.subj != t, (self.subj == t) & (self.part == "calibration")
        test = (self.subj == t) & (self.part == "test")
        args = [self.X[src], self.y[src], self.subj[src], self.X[cal], self.y[cal], self.X[test], t]
        with self.assertRaises(ValueError):                              # target among the sources
            TSALR().fit_predict(*args[:2], np.where(self.subj[src] == "subject_01", t, self.subj[src]), *args[3:])
        y_src = self.y[src].copy()
        y_src[self.subj[src] == "subject_01"] = 0                         # one source loses a class
        with self.assertRaises(ValueError):
            TSALR().fit_predict(args[0], y_src, *args[2:])
        with self.assertRaises(ValueError):                              # calibration with one class
            TSALR().fit_predict(*args[:4], np.zeros(int(cal.sum()), dtype=int), *args[5:])

    def test_tsalr_three_common_classes_and_different_sets(self):
        """API capability (not part of the validated binary evidence): a common set of three classes is accepted;
        a calibration whose class set differs from the sources' is refused before any fit."""
        rng = np.random.default_rng(3)

        def subject(n_per_class, classes):
            X, y = [], []
            for c in classes:
                A = rng.standard_normal((n_per_class, 4, 12)) * (1 + np.eye(4)[c][:, None])
                X.append(A @ A.transpose(0, 2, 1) / 12 + 0.1 * np.eye(4))
                y += [c] * n_per_class
            return np.concatenate(X), np.array(y)

        Xs, ys, ds = [], [], []
        for d in ("subject_01", "subject_02", "subject_03"):
            X, y = subject(10, (0, 1, 2))
            Xs.append(X)
            ys.append(y)
            ds += [d] * len(y)
        X_src, y_src, d_src = np.concatenate(Xs), np.concatenate(ys), np.array(ds)
        X_cal, y_cal = subject(6, (0, 1, 2))
        X_test, _ = subject(4, (0, 1, 2))
        y_hat = TSALR().fit_predict(X_src, y_src, d_src, X_cal, y_cal, X_test, "subject_09")
        self.assertEqual(len(y_hat), len(X_test))
        self.assertTrue(set(y_hat.tolist()) <= {0, 1, 2})
        X_cal2, y_cal2 = subject(6, (0, 1))
        with self.assertRaises(ValueError):
            TSALR().fit_predict(X_src, y_src, d_src, X_cal2, y_cal2, X_test, "subject_09")

    def test_tsalr_test_rows_enter_no_fit_and_rng_restored(self):
        t = "subject_02"
        src, cal = self.subj != t, (self.subj == t) & (self.part == "calibration")
        test = (self.subj == t) & (self.part == "test")
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        p1, p2 = ObservedTSALR(), ObservedTSALR()
        p1.fit_predict(self.X[src], self.y[src], self.subj[src], self.X[cal], self.y[cal], self.X[test], t)
        self.assertTrue(np.array_equal(np.random.get_state()[1], before), "global RNG not restored")
        p2.fit_predict(self.X[src], self.y[src], self.subj[src], self.X[cal], self.y[cal], self.X[test][::-1] * 3, t)
        for k in p1.log[0]["state"]:                   # replacing the test covariances leaves every fitted state equal
            self.assertTrue(np.array_equal(p1.log[0]["state"][k], p2.log[0]["state"][k]), k)


if __name__ == "__main__":
    unittest.main()
