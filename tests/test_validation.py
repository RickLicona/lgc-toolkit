import unittest

import numpy as np

from lgctk import MetadataError, check_partition, segment_runs


class TestSegmentRuns(unittest.TestCase):
    def test_runs_with_composite_identity(self):
        subj = np.array(["a"] * 5 + ["b"] * 4)
        seg = np.array([1, 1, 2, 2, 2, 1, 1, 2, 2])          # block IDs repeat across subjects: allowed
        widx = np.array([0, 1, 0, 1, 2, 0, 1, 0, 1])
        self.assertEqual(segment_runs(subj, seg, widx), [(0, 2), (2, 5), (5, 7), (7, 9)])

    def test_subject_not_contiguous(self):
        with self.assertRaises(MetadataError):
            segment_runs(["a", "b", "a"], [1, 1, 2], [0, 0, 0])

    def test_segment_reappears(self):
        with self.assertRaises(MetadataError):
            segment_runs(["a"] * 3, [1, 2, 1], [0, 0, 1])

    def test_order_within_segment(self):
        with self.assertRaises(MetadataError):
            segment_runs(["a"] * 3, [1, 1, 1], [0, 2, 1])
        with self.assertRaises(MetadataError):
            segment_runs(["a"] * 2, [1, 1], [3, 3])

    def test_order_with_unsigned_and_extreme_integers(self):
        with self.assertRaises(MetadataError):                       # descending uint64 (np.diff would wrap around)
            segment_runs(["s", "s"], [0, 0], np.array([2, 1], dtype=np.uint64))
        with self.assertRaises(MetadataError):
            segment_runs(["s", "s"], [0, 0], np.array([1, 1], dtype=np.uint64))
        self.assertEqual(segment_runs(["s", "s"], [0, 0], np.array([1, 2], dtype=np.uint64)), [(0, 2)])
        i64 = np.iinfo(np.int64)
        self.assertEqual(segment_runs(["s", "s"], [0, 0], np.array([i64.min, i64.max], dtype=np.int64)), [(0, 2)])

    def test_missing_or_malformed(self):
        with self.assertRaises(MetadataError):
            segment_runs(["a", "a"], None, [0, 1])
        with self.assertRaises(MetadataError):
            segment_runs(["a", "a"], [1.0, np.nan], [0, 1])
        with self.assertRaises(MetadataError):
            segment_runs(["a", "a"], [1, 1], [0.0, 1.0])
        with self.assertRaises(MetadataError):
            segment_runs(["a", "a"], [1, 1, 1], [0, 1])
        with self.assertRaises(MetadataError):
            segment_runs([], [], [])


class TestCheckPartition(unittest.TestCase):
    def test_segment_split_between_calibration_and_test(self):
        part = ["calibration", "test"]
        with self.assertRaises(MetadataError):                    # same segment on both sides
            check_partition(["s", "s"], [0, 0], [0, 1], part)
        self.assertEqual(check_partition(["s", "s"], [0, 1], [0, 0], part).tolist(), part)

    def test_each_target_needs_both_sides(self):
        subj = ["a", "a", "b", "b"]
        seg = [1, 2, 1, 2]
        widx = [0, 0, 0, 0]
        part = ["calibration", "test", "calibration", "calibration"]
        with self.assertRaises(MetadataError):
            check_partition(subj, seg, widx, part)                # b has no test segment
        self.assertEqual(len(check_partition(subj, seg, widx, part, targets=["a"])), 4)
        with self.assertRaises(MetadataError):
            check_partition(subj, seg, widx, part, targets=["c"])

    def test_values_and_metadata(self):
        with self.assertRaises(MetadataError):
            check_partition(["s", "s"], [0, 1], [0, 0], ["calibration", "train"])
        with self.assertRaises(MetadataError):
            check_partition(["s", "s"], [0, 1], [0, 0], [0, 1])
        with self.assertRaises(MetadataError):
            check_partition(["s", "s"], [0, 1], [0, 0], ["calibration"])
        with self.assertRaises(MetadataError):
            check_partition(["s", "s"], None, [0, 0], ["calibration", "test"])
        with self.assertRaises(MetadataError):
            check_partition(["s", "s"], [0, 1], [0, 0], None)


if __name__ == "__main__":
    unittest.main()
