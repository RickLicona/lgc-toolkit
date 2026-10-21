# Corrections

Corrections of results obtained with the earlier implementation, `lgc-rct`. Updating this repository does not amend
the published proceedings.

## What went wrong
Two independent routes allowed evaluation-target labels to influence the original pipelines of `lgc-rct`:

1. **FgMDM and sample weights (L1).** In the transductive RCT/FgMDM and LGC-RCT/FgMDM evaluations, assigning zero
   target sample weights did not exclude target labels from FgMDM's internal discriminant fitting step (FGDA uses a
   linear discriminant analysis that does not receive the sample weights) in the implementation used.
2. **Label-inferred segment boundaries (L2).** The DEAP LGC evaluation inferred smoothing-segment boundaries from the
   true target labels (`infer_blocks_from_labels`), allowing those labels to influence the covariance representation
   before classification. The function is generic: any caller using that route would be affected.

The second issue applies to LGC with K > 0 in the DEAP implementation; it does not apply to RCT with K = 0 or to the
Team Metrics experiments, which use recorded block boundaries. Source-only baselines that excluded target samples from
fitting are outside these two mechanisms. The relative contributions of the two mechanisms have not been isolated.

## Affected published results (mean accuracy across subjects, %)
| Paper | Configuration | Published | Corrected | Notes |
|---|---|---|---|---|
| NAT 2026 (Team Metrics) | RCT, alpha / theta | 57.74 / 57.94 | 53.44 / 51.89 | Also includes the filtering correction (fs = 128 Hz); not attributable to L1 alone |
| NAT 2026 | LGC-RCT K10, alpha / theta | 77.73 / 73.91 | 56.85 / 54.61 | Same caveat |
| IEEE MetroXRAINE 2026 (DEAP) | LGC-RCT K10, valence / arousal | 79.52 / 75.51 | 55.70 / 51.00 | L1 + L2 |
| IEEE MetroXRAINE 2026 | RCT, valence / arousal | 64.74 / 62.17 | 54.19 / 50.53 | L1 |

Other papers are not covered by this table.

## How the corrected pipelines are protected
- MDM (which honours sample weights) replaces FgMDM; segments come from recording/trial metadata only.
- Label firewall by construction in this package: evaluation labels never reach the predictive functions
  ([docs/PROTOCOLS.md](docs/PROTOCOLS.md#label-firewall)).
- Evaluation-label invariance tests with positive controls ([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md#tests)).
