"""Predictive pipelines.

Two data-access contracts; evaluation labels are never an argument of either:

- :class:`RCTMDM` (target unlabelled): ``fit_predict(X, domains, target_domain, y_source)`` receives the covariances
  of all subjects and the labels of the SOURCE windows only. See :func:`lgctk.evaluation.loso_unlabelled_target`.
- :class:`TSALR` (target calibration): ``fit_predict(X_source, y_source, source_domains, X_calib, y_calib, X_test,
  target_domain)`` receives the labelled sources, the labelled calibration windows of the target and the test
  covariances WITHOUT labels; test windows enter no fit. See :func:`lgctk.evaluation.loso_calibrated`.
"""
import numpy as np
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import TLCenter, TLClassifier, TLRotate, TLScale, encode_domains
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

MDM_METRIC = {"mean": "logeuclid", "distance": "riemann"}


class RCTMDM:
    """Riemannian re-centering (RCT, ``TLCenter``) + minimum distance to mean (MDM), unlabelled target.

    Data access per fit:

    - every domain (sources and target) is re-centred to its own Riemannian mean using its covariances only;
    - MDM is fitted on the re-centred source windows with their labels (domain weight 1); target windows enter the
      fit with domain weight 0 and a constant placeholder label created here, because ``TLClassifier`` requires a
      label for every sample. MDM honours sample weights, so neither the placeholder nor any target label can change
      the centroids (this is also tested, see ``tests/test_label_invariance.py``);
    - all target windows are predicted.

    Do not replace MDM by FgMDM here: FgMDM's FGDA step fits a discriminant analysis that ignores sample weights.

    Parameters
    ----------
    n_jobs : int, default=1
        Passed to MDM.
    blas_threads : int or None, default=None
        If set, BLAS/OpenMP threads are limited to this number during fit and predict.
    """

    def __init__(self, n_jobs=1, blas_threads=None):
        self.n_jobs = n_jobs
        self.blas_threads = blas_threads

    def _estimator(self):
        return MDM(metric=MDM_METRIC, n_jobs=self.n_jobs)

    def fit_predict(self, X, domains, target_domain, y_source):
        """Fit on all windows (target unlabelled) and predict the target windows.

        Parameters
        ----------
        X : ndarray, shape (n_windows, n_channels, n_channels)
            Covariances of all subjects (sources and target), in the order in which they enter the fit.
        domains : array-like of str, shape (n_windows,)
            Domain (subject) of each window. Must not contain "/" (used by ``encode_domains``).
        target_domain : str
        y_source : array-like of int, shape (n_source_windows,)
            Labels of the source windows, in their order within ``X``.

        Returns
        -------
        ndarray of int, shape (n_target_windows,): predictions for the target windows, in their order within ``X``.
        """
        X = np.asarray(X)
        domains = np.asarray(domains).astype(str)
        if len(domains) != len(X):
            raise ValueError("domains and X differ in length")
        if any("/" in d for d in np.unique(domains)):
            raise ValueError("domain names must not contain '/'")
        is_target = domains == str(target_domain)
        if not is_target.any() or is_target.all():
            raise ValueError("need at least one target and one source window")
        y_source = np.asarray(y_source).astype(int)
        if len(y_source) != int((~is_target).sum()):
            raise ValueError("y_source must have one label per source window")
        if len(np.unique(y_source)) < 2:
            raise ValueError("source labels must contain at least two classes")

        y_fit = np.empty(len(X), dtype=int)
        y_fit[~is_target] = y_source
        y_fit[is_target] = np.unique(y_source)[0]          # placeholder, weight 0 (never a target label)
        weights = {d: 1.0 for d in np.unique(domains)}
        weights[str(target_domain)] = 0.0

        X_enc, y_enc = encode_domains(X, y_fit, domains)
        self.estimator_ = self._estimator()
        self.pipeline_ = make_pipeline(
            TLCenter(target_domain=str(target_domain)),
            TLClassifier(target_domain=str(target_domain), estimator=self.estimator_, domain_weight=weights),
        )
        if self.blas_threads is None:
            self.pipeline_.fit(X_enc, y_enc)
            return self.pipeline_.predict(X[is_target]).astype(int)
        with threadpool_limits(limits=self.blas_threads):
            self.pipeline_.fit(X_enc, y_enc)
            return self.pipeline_.predict(X[is_target]).astype(int)


class TSALR:
    """Tangent space alignment (TSA) + logistic regression, with a labelled calibration part of the target.

    Called TSA-LR: the TSA paper (Bleuze et al., 2022) uses an SVM; logistic regression is this implementation's
    classifier. Composition: ``TangentSpace(metric="riemann")`` -> ``TLCenter`` -> ``TLScale(final_dispersion=1,
    centered_data=True)`` -> ``TLRotate`` (supervised: uses the class means of the labelled calibration windows) ->
    ``TLClassifier(LogisticRegression(solver="lbfgs", max_iter=max_iter))``, every domain with weight 1.

    Data access per call: every fitted stage sees the source windows and the target CALIBRATION windows only; the test
    covariances are used only in ``predict``; test labels are never an argument.

    Reproducibility: ``TLRotate`` fits a one-component PCA without ``random_state`` and scikit-learn may use its
    randomized solver, i.e. the legacy global NumPy RNG. The fit therefore runs with ``np.random.seed(fit_seed)``
    (the previous global state is saved and restored afterwards) and, if ``blas_threads`` is set, fit and predict run
    under ``threadpool_limits(blas_threads)``. Do not run fits of this class concurrently in threads of one process:
    they share the global RNG.

    Parameters
    ----------
    fit_seed : int, default=42
    blas_threads : int or None, default=1
    max_iter : int, default=1000
        Passed to ``LogisticRegression``.
    """

    def __init__(self, fit_seed=42, blas_threads=1, max_iter=1000):
        self.fit_seed = fit_seed
        self.blas_threads = blas_threads
        self.max_iter = max_iter

    def _build_pipeline(self, target_domain, weights):
        self.estimator_ = LogisticRegression(solver="lbfgs", max_iter=self.max_iter)
        return make_pipeline(
            TangentSpace(metric="riemann"),
            TLCenter(target_domain=target_domain),
            TLScale(target_domain=target_domain, final_dispersion=1, centered_data=True),
            TLRotate(target_domain=target_domain),
            TLClassifier(target_domain=target_domain, estimator=self.estimator_, domain_weight=weights),
        )

    def _fit(self, pipeline, X_enc, y_enc):
        state = np.random.get_state()
        try:
            np.random.seed(self.fit_seed)
            pipeline.fit(X_enc, y_enc)
        finally:
            np.random.set_state(state)

    def fit_predict(self, X_source, y_source, source_domains, X_calib, y_calib, X_test, target_domain):
        """Fit on sources + target calibration and predict the target test windows.

        Parameters
        ----------
        X_source : ndarray, shape (n_source, n_channels, n_channels)
        y_source : array-like of int, shape (n_source,)
        source_domains : array-like of str, shape (n_source,)
            Subject of each source window. Must not contain ``target_domain`` nor "/".
        X_calib : ndarray, shape (n_calib, n_channels, n_channels)
            Labelled calibration windows of the target.
        y_calib : array-like of int, shape (n_calib,)
        X_test : ndarray, shape (n_test, n_channels, n_channels)
            Test windows of the target (no labels).
        target_domain : str

        Returns
        -------
        ndarray of int, shape (n_test,): predictions for the test windows, in their order within ``X_test``.
        """
        X_source, X_calib, X_test = np.asarray(X_source), np.asarray(X_calib), np.asarray(X_test)
        y_source, y_calib = np.asarray(y_source).astype(int), np.asarray(y_calib).astype(int)
        source_domains, target_domain = np.asarray(source_domains).astype(str), str(target_domain)
        for name, X in (("X_source", X_source), ("X_calib", X_calib), ("X_test", X_test)):
            if X.ndim != 3 or len(X) == 0:
                raise ValueError(f"{name} must be a non-empty (n, n_channels, n_channels) array")
        if not X_source.shape[1:] == X_calib.shape[1:] == X_test.shape[1:]:
            raise ValueError("X_source, X_calib and X_test differ in matrix size")
        if len(y_source) != len(X_source) or len(source_domains) != len(X_source):
            raise ValueError("y_source and source_domains must have one entry per source window")
        if len(y_calib) != len(X_calib):
            raise ValueError("y_calib must have one label per calibration window")
        if any("/" in d for d in np.unique(source_domains).tolist() + [target_domain]):
            raise ValueError("domain names must not contain '/'")
        if target_domain in set(source_domains.tolist()):
            raise ValueError("target_domain appears among source_domains")
        classes = set(np.unique(y_calib).tolist())        # guards use calibration and source labels only
        if len(classes) < 2:
            raise ValueError("calibration labels must contain at least two classes")
        for d in np.unique(source_domains):
            if set(np.unique(y_source[source_domains == d]).tolist()) != classes:
                raise ValueError(f"source {d!r} does not have the same class set as the calibration {sorted(classes)}")

        X_fit = np.concatenate([X_source, X_calib])
        y_fit = np.concatenate([y_source, y_calib])
        d_fit = np.concatenate([source_domains, np.full(len(X_calib), target_domain)])
        X_enc, y_enc = encode_domains(X_fit, y_fit, d_fit)
        X_test_enc, _ = encode_domains(X_test, np.zeros(len(X_test), dtype=int), np.full(len(X_test), target_domain))
        self.pipeline_ = self._build_pipeline(target_domain, {d: 1.0 for d in np.unique(d_fit)})
        if self.blas_threads is None:
            self._fit(self.pipeline_, X_enc, y_enc)
            return self.pipeline_.predict(X_test_enc).astype(int)
        with threadpool_limits(limits=self.blas_threads):
            self._fit(self.pipeline_, X_enc, y_enc)
            return self.pipeline_.predict(X_test_enc).astype(int)
