"""lgc-toolkit: contextual aggregation of EEG covariance matrices (Local Geometric Consistency, LGC).

Public API (v0.1). Three combinable pieces -- LGC (optional) -> alignment -> classifier:

- :func:`lgctk.lgc.lgc_symmetric`, :func:`lgctk.lgc.lgc_random` -- LGC within recorded segments (mandatory
  subject/segment metadata); skip them to switch LGC off.
- :class:`lgctk.pipelines.RCTMDM` -- Riemannian re-centering (RCT) + MDM with an unlabelled target domain.
- :class:`lgctk.pipelines.TSALR` -- tangent space alignment + logistic regression with a labelled target calibration.
- :func:`lgctk.evaluation.loso_unlabelled_target`, :func:`lgctk.evaluation.loso_calibrated` -- leave-one-subject-out
  evaluators; the only components that see evaluation labels (label firewall).
- :mod:`lgctk.validation` -- metadata and calibration/test partition checks.
- :mod:`lgctk.preprocessing` -- band-pass filtering, sliding windows and Ledoit-Wolf covariances.
- :mod:`lgctk.leakage` -- helpers for the evaluation-label invariance test.
"""
from .evaluation import CalibratedLOSOResult, LOSOResult, loso_calibrated, loso_unlabelled_target
from .lgc import lgc_random, lgc_symmetric, random_neighbourhoods, symmetric_neighbourhoods
from .pipelines import RCTMDM, TSALR
from .validation import MetadataError, check_partition, segment_runs

__version__ = "0.1.0.dev0"

__all__ = [
    "CalibratedLOSOResult",
    "LOSOResult",
    "MetadataError",
    "RCTMDM",
    "TSALR",
    "check_partition",
    "lgc_random",
    "lgc_symmetric",
    "loso_calibrated",
    "loso_unlabelled_target",
    "random_neighbourhoods",
    "segment_runs",
    "symmetric_neighbourhoods",
]
