"""Example 2 -- Team Metrics, labelled calibration: [LGC] + TSA-LR, leave-one-subject-out.

The partition comes from ACQUISITION METADATA, never from labels: recorded blocks 1-6 of the target calibrate (their
labels enter the fit), blocks 7-8 are the test (predicted without labels and scored by the evaluator). The sources
are all other subjects, every block, with labels. Test covariances enter no fitted stage of TSA-LR.

LGC is applied to each source over the whole subject and, for the target, SEPARATELY to the calibration part and to
the test part, so that no neighbourhood mixes calibration with test (for symmetric LGC both give the same result,
because neighbourhoods never leave the block; for random LGC they differ, see docs/REPRODUCIBILITY.md). LGC of the
test part uses the test covariances of the same block (offline preprocessing, not a fit).

    python examples/team_metrics_calibrated_tsa.py --data /path/to/team_metrics.csv
    python examples/team_metrics_calibrated_tsa.py --data ... --lgc none
    python examples/team_metrics_calibrated_tsa.py --data ... --calibration-blocks 1 2 3 4 5 6
"""
import argparse

import numpy as np
import pandas as pd

from lgctk import TSALR, lgc_random, lgc_symmetric, loso_calibrated

from _team_metrics_io import BANDS, features, load


def apply_lgc(covs, subj, seg, pos, kind, k, seed):
    if kind == "symmetric":
        return lgc_symmetric(covs, subj, seg, pos, half_window=k)
    if kind == "random":
        return lgc_random(covs, subj, seg, pos, half_window=k, seed=seed)
    return covs.copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="Team Metrics CSV (private dataset)")
    ap.add_argument("--band", choices=list(BANDS), default="alpha")
    ap.add_argument("--window-sec", type=float, default=4.0, help="window length in seconds (validated: 4)")
    ap.add_argument("--step-sec", type=float, default=2.0, help="window step in seconds (validated: 2, i.e. 50%% overlap)")
    ap.add_argument("--lgc", choices=["symmetric", "random", "none"], default="symmetric")
    ap.add_argument("--half-window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="seed of random LGC (not the TSA fit seed)")
    ap.add_argument("--calibration-blocks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                    help="recorded blocks of the target used for calibration; the others are the test")
    ap.add_argument("--targets", nargs="+", type=int)
    a = ap.parse_args()

    covs, meta = features(load(a.data), a.band, window_sec=a.window_sec, step_sec=a.step_sec)
    subj, seg, pos = meta["subject"].to_numpy(), meta["block"].to_numpy(), meta["window_in_block"].to_numpy()
    partition = np.where(np.isin(seg, a.calibration_blocks), "calibration", "test")     # from metadata only

    X_source = apply_lgc(covs, subj, seg, pos, a.lgc, a.half_window, a.seed)
    X_target = np.empty_like(X_source)
    for part in ("calibration", "test"):
        m = partition == part
        X_target[m] = apply_lgc(covs[m], subj[m], seg[m], pos[m], a.lgc, a.half_window, a.seed)

    targets = None if a.targets is None else [f"subject_{t:02d}" for t in a.targets]
    res = loso_calibrated(X_source, subj, meta["y"].to_numpy(), partition, TSALR(), segment_ids=seg,
                          window_index=pos, window_ids=meta["window_id"].to_numpy(), target_covs=X_target,
                          targets=targets)
    print(pd.DataFrame(res.per_subject).to_string(index=False))
    print(f"mean test accuracy {100 * res.mean_accuracy:.2f}% over {len(res.per_subject)} targets")


if __name__ == "__main__":
    main()
