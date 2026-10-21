"""Synthetic block-structured EEG recordings and the end-to-end paths used by the tests.

A recording is a table per subject (samples x channels) with a recorded ``block`` column and a ``cls`` column (the
class of each sample, constant within a block, alternating between blocks as in Team Metrics). The tests inject fake
labels into ``cls`` -- the data source the pipeline reads -- and never into anything downstream.
"""
import numpy as np
from pyriemann.classification import FgMDM
from pyriemann.transfer import TLCenter, TLClassifier, encode_domains
from sklearn.pipeline import make_pipeline

from lgctk import RCTMDM, TSALR, lgc_random, lgc_symmetric, loso_calibrated, loso_unlabelled_target
from lgctk.leakage import FitSpy, StageSpy, fit_access_check
from lgctk.preprocessing import bandpass, lw_covariances, sliding_windows

FS, LO, HI = 32, 4, 12
WIN, STEP = 4 * FS, 2 * FS
N_CHANNELS, N_SUBJECTS, N_BLOCKS, BLOCK_SAMPLES = 4, 4, 4, 30 * FS
K = 3
SEED = 42
CAL_BLOCKS = (1, 2)          # calibrated path: recorded blocks 1-2 calibrate, 3-4 test (from metadata, never from y)


def make_recordings(seed=7):
    rng = np.random.default_rng(seed)
    recs = {}
    for s in range(1, N_SUBJECTS + 1):
        mix = np.eye(N_CHANNELS) + 0.3 * rng.standard_normal((N_CHANNELS, N_CHANNELS))
        eeg, block, cls = [], [], []
        for b in range(1, N_BLOCKS + 1):
            c = (b + s) % 2
            scale = np.ones(N_CHANNELS)
            scale[c] = 2.0
            drift = 1 + 0.5 * np.sin(np.linspace(0, 3, BLOCK_SAMPLES))[:, None]
            src = rng.standard_normal((BLOCK_SAMPLES, N_CHANNELS)) * scale * drift
            eeg.append(src @ mix.T)
            block.append(np.full(BLOCK_SAMPLES, b))
            cls.append(np.full(BLOCK_SAMPLES, c))
        recs[f"subject_{s:02d}"] = dict(eeg=np.concatenate(eeg), block=np.concatenate(block), cls=np.concatenate(cls))
    return recs


def block_labels(r):
    """Recorded blocks of one subject (sorted) and the class of each block, read from the class column."""
    blocks = np.array(sorted(np.unique(r["block"])))
    return blocks, np.array([int(r["cls"][r["block"] == b][0]) for b in blocks])


def with_target_block_labels(recs, target, labels_per_block):
    """Copy of the recordings whose target class column carries ``labels_per_block`` (source-level injection)."""
    out = {s: dict(r) for s, r in recs.items()}
    r = dict(recs[target])
    cls = np.array(r["cls"], copy=True)
    for b, v in zip(block_labels(r)[0], labels_per_block):
        cls[r["block"] == b] = v
    r["cls"] = cls
    out[target] = r
    return out


def featurize(recs, segments="recorded", window_labels=None):
    """Covariances + metadata for all subjects (sorted). ``segments='from_labels'`` is the deliberate L2 leak.
    ``window_labels`` = (subject, per-window labels): optional window-level injection, right after windowing and
    before anything consumes the labels."""
    covs, subj, seg, widx, wid, y = [], [], [], [], [], []
    for s in sorted(recs):
        r = recs[s]
        for b in sorted(np.unique(r["block"])):
            rows = r["block"] == b
            sig = bandpass(r["eeg"][rows].T, LO, HI, FS, axis=1)
            w, starts = sliding_windows(sig, WIN, STEP)
            covs.append(lw_covariances(w))
            label = int(r["cls"][rows][0])
            for k, st in enumerate(starts):
                subj.append(s)
                seg.append(b)
                widx.append(k)
                wid.append(f"{s}|b{b}|{st}")
                y.append(label)
    covs, subj, seg, y = np.concatenate(covs), np.array(subj), np.array(seg), np.array(y)
    if window_labels is not None:
        y = y.copy()
        y[subj == window_labels[0]] = window_labels[1]
    if segments == "from_labels":
        seg = segments_from_labels(y, subj)
        widx = np.arange(len(y))
    return covs, subj, seg, np.array(widx), np.array(wid), y


def segments_from_labels(y, subj):
    """The L2 mechanism (re-implemented as a positive control): a new segment whenever the label changes."""
    new = np.r_[True, (y[1:] != y[:-1]) | (subj[1:] != subj[:-1])]
    return np.cumsum(new)


class ObservedRCTMDM(RCTMDM):
    """RCTMDM that records, per fit, the labels/weights received by MDM and its centroids."""

    def __init__(self):
        super().__init__()
        self.log = []

    def _estimator(self):
        est = super()._estimator()
        self._spy = FitSpy(est)
        return est

    def fit_predict(self, X, domains, target_domain, y_source):
        y_hat = super().fit_predict(X, domains, target_domain, y_source)
        call = self._spy.calls[-1]
        self.log.append(dict(target=target_domain, y_fed=call["y"], w_fed=call["sample_weight"],
                             state={"covmeans_": np.array(self.estimator_.covmeans_)}))
        return y_hat


def apply_lgc(covs, subj, seg, widx, lgc, half_window, seed=SEED):
    if lgc == "symmetric":
        return lgc_symmetric(covs, subj, seg, widx, half_window=half_window)
    if lgc == "random":
        return lgc_random(covs, subj, seg, widx, half_window=half_window, seed=seed)
    if lgc == "none":
        return covs
    raise ValueError(lgc)


def run_package_path(recs, target, segments="recorded", window_labels=None, half_window=K,
                     fit_order="sources_then_target", lgc="symmetric", seed=SEED):
    """End-to-end package path for one target: recordings -> windows -> segments -> LGC -> RCT+MDM -> scoring.
    ``half_window=0`` is the RCT configuration without LGC."""
    covs, subj, seg, widx, wid, y = featurize(recs, segments, window_labels)
    X_lgc = apply_lgc(covs, subj, seg, widx, lgc, half_window, seed)
    pipe = ObservedRCTMDM()
    res = loso_unlabelled_target(X_lgc, subj, y, pipe, window_ids=wid, targets=[target], fit_order=fit_order)
    log = pipe.log[0]
    return dict(y_hat=res.y_pred, window_ids=res.window_ids, y_true_seen_by_evaluator=res.y_true, X_lgc=X_lgc,
                segments=seg, state=log["state"], y_fed=log["y_fed"], w_fed=log["w_fed"], subj=subj)


def run_original_fgmdm_path(recs, target, segments="recorded", window_labels=None):
    """Positive control C+1 (L1): the original lgc-rct construction -- target TRUE labels in the fit with weight 0,
    FgMDM classifier (its FGDA/LDA step ignores sample weights)."""
    covs, subj, seg, widx, wid, y = featurize(recs, segments, window_labels)
    X_lgc = lgc_symmetric(covs, subj, seg, widx, half_window=K)
    is_t = subj == target
    order = np.r_[np.flatnonzero(~is_t), np.flatnonzero(is_t)]
    X, d, yy = X_lgc[order], subj[order], y[order]
    weights = {s: 1.0 for s in np.unique(d)}
    weights[target] = 0.0
    est = FgMDM(metric="riemann", n_jobs=1)
    X_enc, y_enc = encode_domains(X, yy, d)
    pipe = make_pipeline(TLCenter(target_domain=target),
                         TLClassifier(target_domain=target, estimator=est, domain_weight=weights))
    pipe.fit(X_enc, y_enc)
    y_hat = pipe.predict(X[d == target]).astype(int)
    state = {"fgda_W": np.array(est._fgda._W), "mdm_covmeans_": np.array(est._mdm.covmeans_)}
    return dict(y_hat=y_hat, X_lgc=X_lgc, segments=seg, state=state)


# ---- calibrated path (TSA-LR) ---------------------------------------------------------------------------------------
TSA_STAGES = 5


def calibration_partition(subj, seg):
    """Partition from recorded metadata: blocks in CAL_BLOCKS calibrate, the others test (for every subject)."""
    return np.where(np.isin(seg, CAL_BLOCKS), "calibration", "test")


def target_representation(covs, subj, seg, widx, partition, lgc, half_window, seed=SEED):
    """Representation of every window as TARGET: LGC applied separately to the calibration and test parts."""
    if lgc == "none":
        return covs.copy()
    out = np.empty_like(covs)
    for part in ("calibration", "test"):
        m = partition == part
        out[m] = apply_lgc(covs[m], subj[m], seg[m], widx[m], lgc, half_window, seed)
    return out


class ObservedTSALR(TSALR):
    """TSALR that records, per fit, the rows each stage received and the fitted state of every stage."""

    def __init__(self):
        super().__init__()
        self.log = []

    def _build_pipeline(self, target_domain, weights):
        pipe = super()._build_pipeline(target_domain, weights)
        self.spy = StageSpy(pipe, extra=[("extra:classifier", self.estimator_)])
        return pipe

    def fit_predict(self, X_source, y_source, source_domains, X_calib, y_calib, X_test, target_domain):
        y_hat = super().fit_predict(X_source, y_source, source_domains, X_calib, y_calib, X_test, target_domain)
        steps = dict(self.pipeline_.steps)
        state = {"coef_": np.array(self.estimator_.coef_), "intercept_": np.array(self.estimator_.intercept_),
                 "reference_": np.array(steps["tangentspace"].reference_)}
        for name, attr in (("tlcenter", "centers_"), ("tlscale", "scales_"), ("tlrotate", "rotations_")):
            for d, v in sorted(getattr(steps[name], attr).items()):
                state[f"{name}.{attr}[{d}]"] = np.array(v)
        self.log.append(dict(calls=self.spy.calls, state=state))
        return y_hat


class LeakyUnsupervisedTSALR(ObservedTSALR):
    """NEGATIVE FIXTURE (deliberate defect): the unsupervised stages (tangent space, centring, scaling) are fitted on
    the fit rows PLUS the test covariances, without any test label; the rest of the pipeline is fitted as usual."""

    def fit_predict(self, X_source, y_source, source_domains, X_calib, y_calib, X_test, target_domain):
        self._x_test, self._target = np.asarray(X_test), str(target_domain)
        return super().fit_predict(X_source, y_source, source_domains, X_calib, y_calib, X_test, target_domain)

    def _fit(self, pipeline, X_enc, y_enc):
        X_t, y_t = encode_domains(self._x_test, np.zeros(len(self._x_test), dtype=int),
                                  np.full(len(self._x_test), self._target))
        state = np.random.get_state()
        try:
            np.random.seed(self.fit_seed)
            Z = pipeline[:3].fit_transform(np.concatenate([X_enc, X_t]), np.concatenate([y_enc, y_t]))
            pipeline[3:].fit(Z[:len(X_enc)], y_enc)
        finally:
            np.random.set_state(state)


def leaky_labelled_loso(covs, subj, y, partition, pipeline, *, segment_ids, window_index, window_ids, target_covs,
                        targets):
    """POSITIVE CONTROL (deliberate defect of an evaluator): the target TEST windows are appended, WITH their labels,
    to the calibration part handed to the pipeline."""
    t = targets[0]
    is_t, src = subj == t, subj != t
    cal, test = is_t & (partition == "calibration"), is_t & (partition == "test")
    X_cal = np.concatenate([target_covs[cal], target_covs[test]])
    y_cal = np.concatenate([y[cal], y[test]])
    y_hat = pipeline.fit_predict(covs[src], y[src], subj[src], X_cal, y_cal, target_covs[test], t)
    return dict(window_ids=window_ids[test], y_true=y[test], y_pred=np.asarray(y_hat).astype(int))


def run_calibrated_path(recs, target, window_labels=None, lgc="symmetric", half_window=K, seed=SEED,
                        pipeline_cls=ObservedTSALR, evaluator="package"):
    """End-to-end calibrated path for one target: recordings -> windows -> metadata partition -> LGC (sources: whole
    subject; target: per partition) -> TSA-LR -> scoring. Returns predictions, representations, fitted state and the
    data-access check computed from the harness's own partition."""
    covs, subj, seg, widx, wid, y = featurize(recs, window_labels=window_labels)
    partition = calibration_partition(subj, seg)
    X_src = apply_lgc(covs, subj, seg, widx, lgc, half_window, seed)
    X_tgt = target_representation(covs, subj, seg, widx, partition, lgc, half_window, seed)
    pipe = pipeline_cls()
    kwargs = dict(segment_ids=seg, window_index=widx, window_ids=wid, target_covs=X_tgt, targets=[target])
    if evaluator == "package":
        res = loso_calibrated(X_src, subj, y, partition, pipe, **kwargs)
        out = dict(y_hat=res.y_pred, window_ids=res.window_ids, y_true_seen_by_evaluator=res.y_true)
    else:
        r = leaky_labelled_loso(X_src, subj, y, partition, pipe, **kwargs)
        out = dict(y_hat=r["y_pred"], window_ids=r["window_ids"], y_true_seen_by_evaluator=r["y_true"])
    log = pipe.log[0]
    is_t = subj == target
    cal, test = is_t & (partition == "calibration"), is_t & (partition == "test")
    access = fit_access_check(log["calls"], np.concatenate([X_src[~is_t], X_tgt[cal]]), X_tgt[test], TSA_STAGES)
    out.update(X_src=X_src, X_tgt=X_tgt, partition=partition, state=log["state"], access=access,
               y_fed=log["calls"][-1]["y"], fit_rows=log["calls"][0]["X"])
    return out
