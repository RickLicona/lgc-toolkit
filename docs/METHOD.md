# Method

## Local Geometric Consistency (LGC)
Let $C_1, \dots, C_N$ be the covariance matrices of one subject's windows in temporal order, and let $s(i)$ be the
recorded segment (block or trial) of window $i$. LGC replaces each matrix by the Riemannian (affine-invariant,
Fréchet) mean of a neighbourhood $\mathcal{N}(i)$ of windows of the same segment:

$$\bar C_i = \operatorname*{arg\,min}_{P \succ 0} \sum_{j \in \mathcal{N}(i)} \delta_R^2(P, C_j),$$

computed with `pyriemann.utils.mean.mean_riemann` (default tolerance and iterations).

### Symmetric neighbourhood (`lgc_symmetric`)
For a half-window $K \ge 0$,

$$\mathcal{N}_K(i) = \{\, j : |j - i| \le K,\ s(j) = s(i) \,\},$$

that is, up to $K$ windows on each side, clipped at the segment edges (at most $2K + 1$ windows, fewer near the
edges). $K = 0$ leaves the matrices unchanged. $K$ is a free parameter; $K = 10$ is the value of the validated
configurations.

**What K means in time.** K is a radius counted in **windows**. With a window step $\Delta t$, the start samples of
the symmetric neighbours reach up to $\pm K \Delta t$ around the window's own start, where the segment allows it:
K = 10 with a 2 s step gives $\pm 20$ s; the window length is added to the total span of signal involved. For
**random** neighbourhoods, K only sets the number of neighbours (the size of the symmetric neighbourhood at that
position); it does **not** limit their distance to $\pm K \Delta t$: they may come from anywhere in the same
segment.

### Random neighbourhood (`lgc_random`)
For every window $i$, the neighbourhood has the size $|\mathcal{N}_K(i)|$ of the symmetric one at that position; it
contains $i$ and $|\mathcal{N}_K(i)| - 1$ other windows of the same segment drawn uniformly without replacement. A
fresh `numpy.random.default_rng(seed)` is created for each subject in each call and visits the subject's windows in
order; no number is drawn when no extra member is needed. The seed is mandatory. The draws depend on which windows
enter the call (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

### Properties of both
- **No labels, no fitted state.** The functions receive covariances and acquisition metadata only.
- **Segment-bounded.** Neighbourhoods never cross a (subject, segment) boundary; segments come from recording
  metadata ([PROTOCOLS.md](PROTOCOLS.md#segments-and-metadata)).
- **Offline.** Neighbourhoods may include future windows.
- **Order-preserving.** Output row $i$ corresponds to input row $i$; data are validated, never reordered.

## Alignment and classifiers
### RCT + MDM (`RCTMDM`)
```
[LGC] ──► RCT: every subject re-centred to its own Riemannian mean   (TLCenter; target covariances without labels)
      ──► MDM fitted on the source subjects                         (log-Euclidean mean, Riemannian distance)
      ──► predict every target window ──► scoring by the evaluator only
```
### TSA-LR (`TSALR`)
```
[LGC] ──► tangent space at the Riemannian mean of the fit set        (TangentSpace)
      ──► per-domain centring and scaling (dispersion 1)             (TLCenter, TLScale)
      ──► rotation of the sources towards the target                 (TLRotate; uses the class means of the
                                                                       labelled target calibration)
      ──► logistic regression on sources + target calibration        (all domains weight 1; lbfgs, max_iter 1000)
      ──► predict the target test windows ──► scoring by the evaluator only
```
Every fitted stage receives the sources and the target calibration windows only; the test covariances are used only
in prediction. The TSA paper (Bleuzé et al., 2022) uses an SVM; this implementation uses logistic regression and is
called TSA-LR.

See [PROTOCOLS.md](PROTOCOLS.md) for what each stage may access.

## Preprocessing (optional utilities)
The core functions receive covariance matrices: any preprocessing may be used before them. `lgctk.preprocessing`
provides the utilities used by the examples and by the validated configurations; they are common choices, not
universal requirements of EEG decoding.

```python
from lgctk.preprocessing import bandpass, sliding_windows, lw_covariances

# ONE recorded segment (block or trial), shape (channels, samples); the parameters are the scientist's choice
filtered = bandpass(segment, lo=8, hi=13, fs=128, axis=1, order=4)
windows, starts = sliding_windows(filtered, window_len=512, step_len=256)    # 4 s every 2 s at 128 Hz
covs = lw_covariances(windows)
# then: per-window metadata -> optional LGC -> RCTMDM or TSALR -> evaluation
```
- `bandpass(signal, lo, hi, fs, axis, order=4)`: Butterworth design `scipy.signal.butter(order, ...)` applied
  forwards and backwards with `scipy.signal.filtfilt` (zero phase, offline). The design order is not the order of the
  effective forward-backward response. `fs` and `axis` are mandatory.
- `sliding_windows(signal, window_len, step_len)`: windows of a `(channels, samples)` array, in samples
  (`window_len = window_sec * fs`, `step_len = step_sec * fs`; 4 s every 2 s overlap by 50 %), float32, with their
  start samples; only windows that fit entirely in the array are returned.
- `lw_covariances(windows)`: Ledoit-Wolf shrunk covariances.

The functions know nothing about acquisition boundaries: **apply them to one segment at a time** so that no filter
transient and no window crosses a segment. Band, `fs`, window length, step and K must be chosen for each dataset; a
different combination does not inherit the validated figures.

## Interpretation
LGC is a representation step. Gains reported for one dataset and protocol are not guarantees for another, and mean
gains across subjects do not imply benefit for every subject. In block-structured data, segment boundaries carry
protocol information (see [PROTOCOLS.md](PROTOCOLS.md#limitations)).

## Not in this release
Past-only (causal) neighbourhoods, other means, and RCT + MDM with a labelled target calibration.
