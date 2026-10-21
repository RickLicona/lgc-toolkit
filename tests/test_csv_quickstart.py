"""The CSV quick-start example: window counts, partitions per segment and the input checks of its adapter
(synthetic recording; no real EEG)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:                       # pandas is only needed by the examples
    pd = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

FS, CHANNELS = 128, ["Fp1", "Fp2"]


@unittest.skipIf(pd is None, "pandas not installed")
class TestCSVQuickstart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import csv_quickstart as qs
        cls.qs = qs
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "eeg_dataset.csv"
            qs.make_synthetic_csv(path)
            cls.data = pd.read_csv(path)

    def prepare(self, data):
        return self.qs.prepare(data, CHANNELS, FS)

    def test_windows_metadata_and_routes(self):
        X, meta, window_ids = self.prepare(self.data)
        self.assertEqual(X.shape, (3 * 8 * 14, 2, 2))              # 30 s segments, 4 s windows every 2 s
        self.assertEqual(list(meta["position"][:14]), list(range(14)))
        self.assertEqual(len(np.unique(window_ids)), len(meta))
        for route in self.qs.ROUTES:
            res = self.qs.run(X, meta, window_ids, route)
            self.assertEqual(len(res.per_subject), 3)
            self.assertEqual(len(res.y_true), 336 if route == "unlabelled" else 84)

    def test_partitions_are_constant_per_segment(self):
        _, meta, _ = self.prepare(self.data)
        for partition in (self.qs.partition_by_acquisition_order(meta), self.qs.partition_stratified(meta)):
            per_segment = pd.Series(partition).groupby([meta["subject"], meta["segment"]]).nunique()
            self.assertTrue((per_segment == 1).all())
            test_segments = meta[partition == "test"].groupby("subject")["segment"].nunique()
            self.assertEqual(test_segments.tolist(), [2, 2, 2])

    def test_acquisition_order_is_file_order_not_name_order(self):
        data = self.data.copy()
        rename = {f"trial_{k:02d}": f"z{9 - k}" for k in range(1, 9)}   # names sort opposite to file order
        data["segment"] = data["segment"].map(rename)
        _, meta, _ = self.prepare(data)
        partition = self.qs.partition_by_acquisition_order(meta)
        first = meta[meta["subject"] == "S01"]
        test_names = sorted(first.loc[partition[first.index] == "test", "segment"].unique())
        self.assertEqual(test_names, ["z1", "z2"])                       # trials 7 and 8 of the file

    def test_stratified_split_fails_instead_of_falling_back(self):
        _, meta, _ = self.prepare(self.data)
        meta = meta.copy()
        one_class_trial = (meta["subject"] == "S01") & (meta["segment"] != "trial_02")
        meta.loc[one_class_trial, "label"] = 0                           # S01: a single trial of class 1
        with self.assertRaisesRegex(ValueError, "S01"):
            self.qs.partition_stratified(meta)

    def test_input_checks(self):
        d = self.data
        cases = {
            "lacks columns": d.drop(columns="segment"),
            "no rows": d.iloc[:0],
            "missing values": d.assign(segment=d["segment"].where(d.index != 5)),
            "not numeric": d.assign(Fp1=d["Fp1"].astype(str)),
            "non-finite": d.assign(Fp2=d["Fp2"].where(d.index != 7, np.inf)),
            "integers": d.assign(label=d["label"] + 0.5),
            "more than one label": d.assign(label=d["label"].where(d.index != 0, 1)),
            "each \\(subject, segment\\) must be contiguous": pd.concat([d.iloc[:100], d.iloc[4000:4100], d.iloc[100:200]]),
            "each subject must be contiguous": pd.concat([d[d["subject"] == "S01"].iloc[:3840],
                                                         d[d["subject"] == "S02"].iloc[:3840],
                                                         d[d["subject"] == "S01"].iloc[3840:7680]]),
            "shorter than one window": d.iloc[:FS],
        }
        for message, data in cases.items():
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare(data.reset_index(drop=True))
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            self.qs.prepare(d, CHANNELS, fs=20)


if __name__ == "__main__":
    unittest.main()
