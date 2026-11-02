import numpy as np
from typing import Dict
from sklearn.metrics import roc_curve

def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))

def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0))))

def fit_threshold_and_scale(labels: np.ndarray, raw_probs: np.ndarray) -> Dict[str, float]:
    """
    Computes the optimal ROC operating threshold using Youden's J-statistic
    and calculates a log-odds scaling factor for probability calibration.
    """
    labels = np.asarray(labels, dtype=int)
    raw_probs = np.asarray(raw_probs, dtype=float)

    real_probs = raw_probs[labels == 0]
    fake_probs = raw_probs[labels == 1]

    if len(real_probs) > 0 and len(fake_probs) > 0:
        max_real = float(np.max(real_probs))
        min_fake = float(np.min(fake_probs))
        if min_fake > max_real:
            # Clean separation: place threshold at midpoint
            optimal_threshold = 0.5 * (max_real + min_fake)
        else:
            fpr, tpr, thresholds = roc_curve(labels, raw_probs)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            optimal_threshold = float(np.clip(thresholds[best_idx], 0.05, 0.95))

        # Scaling factor so separation maps smoothly across confidence range
        mean_diff = abs(_logit(float(np.mean(fake_probs))) - _logit(float(np.mean(real_probs))))
        scale = float(np.clip(2.8 / max(mean_diff, 0.05), 1.0, 15.0))
    else:
        optimal_threshold = 0.5
        scale = 1.0

    return {
        "optimal_threshold": round(optimal_threshold, 6),
        "scale": round(scale, 4)
    }

def apply_calibration(raw_prob: float, calib_params: Dict[str, float]) -> float:
    """
    Maps raw model probability into a calibrated [0, 1] probability
    centered at 0.50 when raw_prob == optimal_threshold.
    """
    tau = calib_params.get("optimal_threshold", 0.5)
    scale = calib_params.get("scale", 1.0)
    z_raw = _logit(raw_prob)
    z_tau = _logit(tau)
    return _sigmoid(scale * (z_raw - z_tau))
