"""Example 1 -- Team Metrics, target without labels: [LGC] + RCT + MDM, leave-one-subject-out.

Data access: every target's covariances enter the unsupervised re-centring (RCT) without labels; MDM is fitted on the
labelled sources only; all target windows are predicted and scored by the evaluator. Symmetric and random LGC use
neighbouring windows of the same recorded block, future ones included: results are offline.

    python examples/team_metrics_transductive.py --data /path/to/team_metrics.csv
    python examples/team_metrics_transductive.py --data ... --band theta --lgc random --seed 42 --half-window 10
    python examples/team_metrics_transductive.py --data ... --lgc none
"""
import argparse

import pandas as pd

from lgctk import RCTMDM, lgc_random, lgc_symmetric, loso_unlabelled_target

from _team_metrics_io import BANDS, features, load


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="Team Metrics CSV (private dataset)")
    ap.add_argument("--band", choices=list(BANDS), default="alpha")
    ap.add_argument("--window-sec", type=float, default=4.0, help="window length in seconds (validated: 4)")
    ap.add_argument("--step-sec", type=float, default=2.0, help="window step in seconds (validated: 2, i.e. 50%% overlap)")
    ap.add_argument("--lgc", choices=["symmetric", "random", "none"], default="symmetric")
    ap.add_argument("--half-window", type=int, default=10, help="LGC radius K in windows")
    ap.add_argument("--seed", type=int, default=42, help="seed of random LGC")
    ap.add_argument("--targets", nargs="+", type=int, help="target subject numbers (default: all)")
    a = ap.parse_args()

    covs, meta = features(load(a.data), a.band, window_sec=a.window_sec, step_sec=a.step_sec)
    subj, seg, pos = meta["subject"].to_numpy(), meta["block"].to_numpy(), meta["window_in_block"].to_numpy()
    if a.lgc == "symmetric":
        covs = lgc_symmetric(covs, subj, seg, pos, half_window=a.half_window)
    elif a.lgc == "random":
        covs = lgc_random(covs, subj, seg, pos, half_window=a.half_window, seed=a.seed)
    window_ids = (meta["subject"] + "|" + meta["window_id"]).to_numpy()
    targets = None if a.targets is None else [f"subject_{t:02d}" for t in a.targets]
    res = loso_unlabelled_target(covs, subj, meta["y"].to_numpy(), RCTMDM(), window_ids=window_ids, targets=targets)
    print(pd.DataFrame(res.per_subject).to_string(index=False))
    print(f"mean accuracy {100 * res.mean_accuracy:.2f}% over {len(res.per_subject)} targets")


if __name__ == "__main__":
    main()
