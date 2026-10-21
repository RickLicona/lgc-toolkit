"""The Team Metrics example extraction, now built on lgctk.preprocessing.sliding_windows, gives the same windows,
start samples and covariances as the previous explicit loop (synthetic recording; no real EEG)."""
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:                       # pandas is only needed by the examples
    pd = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


def previous_loop(df, band, io):
    """The previous extraction of examples/_team_metrics_io.features, copied literally."""
    from lgctk.preprocessing import bandpass, lw_covariances
    lo, hi = io.BANDS[band]
    win, step = io.WINDOW_SEC * io.FS, io.STEP_SEC * io.FS
    covs, meta = [], []
    for subject, df_subj in df.groupby("subject"):
        X = []
        for block in sorted(df_subj["block"].unique()):
            rows = df_subj[df_subj["block"] == block]
            label = int(rows["class"].iloc[0])
            sig = bandpass(rows[io.CHANNELS].to_numpy(), lo, hi, fs=io.FS, axis=0)
            for k, start in enumerate(range(0, sig.shape[0] - win + 1, step)):
                X.append(sig[start:start + win, :])
                meta.append((io.domain(subject), int(block), k, start, label))
        covs.append(lw_covariances(np.transpose(np.stack(X).astype(np.float32), (0, 2, 1))))
    return np.concatenate(covs), meta


@unittest.skipIf(pd is None, "pandas not installed")
class TestTeamMetricsExampleExtraction(unittest.TestCase):
    def test_same_windows_starts_and_covariances(self):
        import _team_metrics_io as io
        rng = np.random.default_rng(5)
        frames = []
        for s in (6, 8):
            for b in (1, 2, 3):
                n = int(rng.integers(10 * io.FS, 14 * io.FS))            # unequal block lengths
                df = pd.DataFrame(rng.standard_normal((n, len(io.CHANNELS))), columns=io.CHANNELS)
                df["time"], df["block"], df["class"], df["subject"] = np.arange(n) / io.FS, b, b % 2, s
                frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        covs_old, meta_old = previous_loop(df, "alpha", io)
        covs_new, meta_new = io.features(df, "alpha")
        self.assertTrue(np.array_equal(covs_old, covs_new))
        self.assertEqual(meta_old, list(meta_new[["subject", "block", "window_in_block", "start_sample", "y"]]
                                        .itertuples(index=False, name=None)))


if __name__ == "__main__":
    unittest.main()
