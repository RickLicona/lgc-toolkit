# Evaluation protocols

Cross-subject transfer within each dataset (leave-one-subject-out). Each protocol is described by **what each stage
may access**.

## Segments and metadata
LGC averages each window with neighbours of the same **segment**, so segment boundaries define the context LGC may
use. They must be correct and independent of the labels.
- `subject_ids`, `segment_ids` and the window order (`window_index`) are **mandatory**. The package validates them,
  never infers segments from labels and never reorders data.
- Segment identifiers come from **acquisition metadata**: the recorded block or the trial. Receiving `segment_ids`
  does not prove that they are independent of the labels; computing them from `y` reintroduces label leakage.
- The identity of a segment is the pair (subject, segment); segments never cross subjects. Symmetric and random LGC
  draw neighbours from the window's own segment only.
- Window identifiers are unique: across the whole dataset for `loso_unlabelled_target`, within each subject for
  `loso_calibrated`.

## Target without labels (`RCTMDM` + `loso_unlabelled_target`)
| Stage | Source subjects | Target subject |
|---|---|---|
| Segmentation, covariances, LGC | signals + recorded segment IDs | signals + recorded segment IDs |
| Re-centering (RCT, unsupervised) | covariances | covariances (no labels) |
| Classifier fit (MDM) | covariances + labels | **not used for fitting** (label firewall) |
| Prediction | — | all target windows |
| Scoring | — | target labels, seen **only** by the evaluator |

## Labelled target calibration (`TSALR` + `loso_calibrated`)
| Stage | Source subjects | Target calibration windows | Target test windows |
|---|---|---|---|
| Segmentation, covariances, LGC | whole subject | calibration part only | test part only (offline: uses test covariances of the same segment; not a fit) |
| Tangent space, centring, scaling | covariances | covariances | **not used** |
| Rotation (supervised) | covariances + labels | covariances + labels | **not used** |
| Classifier fit (logistic regression) | covariances + labels | covariances + labels | **not used** |
| Prediction | — | — | every test window |
| Scoring | — | — | test labels, seen **only** by the evaluator |

### The partition is the user's decision
`partition` gives `"calibration"` or `"test"` for every window. It is part of the benchmark design, chosen by the
user before any evaluation, and is validated by `lgctk.validation.check_partition` before any fit:
- it is constant inside every recorded (subject, segment): a segment is never split between calibration and test;
- every evaluated target has at least one calibration segment and one test segment;
- window identifiers are unique within each subject.

The partition array covers every window, so subjects used only as sources also need valid values (`"calibration"` or
`"test"`); for them the values are not used, and only evaluated targets must have both sides.

Recommended practice, shown in the examples:
- **Partition from metadata when the design allows it** (e.g. Team Metrics: recorded blocks 1–6 calibrate, 7–8
  test).
- **A stratified partition is a benchmark construction.** Stratifying by class (e.g. DEAP: a quarter of each class's
  trials to test) uses the per-trial labels of the target, test trials included. It is done once, outside the
  predictive function, and then **frozen**: it is never recomputed, e.g. when an invariance test injects fake labels.
  Its seed is a separate parameter from the LGC seed and the TSA fit seed.
- **Transform the target's parts separately.** Apply LGC to the target's calibration windows and to its test windows
  in two separate calls and pass the result as `target_covs`, so that no neighbourhood mixes calibration with test
  (sources: LGC over the whole subject). With symmetric LGC both give the same matrices; with random LGC they do not
  (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

The predictive function's guards use calibration and source labels only (every source and the calibration must have
the same class set); test labels never accept, exclude or configure a fold.

## Label firewall
The evaluation labels are not passed to the predictive functions at all; the design does not rely on sample or domain
weights to hide them.
- `lgc_symmetric` and `lgc_random` have no labels argument.
- `RCTMDM.fit_predict` receives the labels of the source windows only. Target windows get a constant placeholder with
  domain weight 0, created inside the pipeline; MDM honours the weights. FgMDM, whose discriminant step ignores the
  sample weights, is not used.
- `TSALR.fit_predict` receives source and calibration labels and the test covariances without labels.
- The evaluators (`loso_unlabelled_target`, `loso_calibrated`) are the only components holding the evaluation labels.

The test suite checks this: see [REPRODUCIBILITY.md](REPRODUCIBILITY.md#tests).

## Limitations
- **Offline.** Symmetric and random neighbourhoods include future windows, and the band-pass filter of
  `lgctk.preprocessing` is zero phase (forward and backward). A causal use would need a past-only neighbourhood and a
  causal pipeline, which are not part of this release.
- **Block structure (Team Metrics).** Each recorded block holds a single experimental condition, so knowing the
  block boundaries already tells which windows share a condition. This is a property of the protocol, not a leak an
  invariance test can detect; state it when reporting LGC results on block-structured data.
- **DEAP.** Participants watched the same 40 videos; cross-subject results are transfer with shared stimuli, not
  generalisation to unseen videos.

## Validated configurations
| Dataset | Band | Configurations |
|---|---|---|
| Team Metrics (workload, 34 participants) | alpha, theta | RCT + MDM; symmetric LGC K=10 + RCT + MDM |
| DEAP (emotion, 32 participants), valence and arousal | beta (15–36 Hz) | RCT + MDM; symmetric LGC K=10 + RCT + MDM |
| Team Metrics | alpha | random LGC K=10 seed 42 + RCT + MDM |
| DEAP, valence and arousal | beta (15–36 Hz) | random LGC K=10 seed 42 + RCT + MDM |
| Team Metrics, blocks 1–6 calibrate, 7–8 test | alpha | TSA-LR; symmetric LGC K=10 + TSA-LR |
| DEAP, valence and arousal, frozen stratified split | beta (15–36 Hz) | TSA-LR; symmetric LGC K=10 + TSA-LR; random LGC K=10 seed 42 + TSA-LR |

Evidence and scope of every row: [VALIDATION.md](../VALIDATION.md).

Window length 4 s, step 2 s, fs 128 Hz, 32 channels, Ledoit-Wolf covariances. DEAP: threshold rating ≥ 5, 3 s
baseline retained.
