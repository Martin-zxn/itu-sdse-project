"""Inference-log drift detection matching the deployment notebook contract."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
from scipy.stats import ks_2samp

DEFAULT_DAYS = 7
KS_PVALUE_THRESHOLD = 0.05
MEAN_DRIFT_THRESHOLD = 0.1


def load_inference_logs(
    log_path: Union[str, Path] = "artifacts/inference_log.json",
):
    """Load a request log; a missing file is the normal no-traffic case."""
    path = Path(log_path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as log_file:
        logs = json.load(log_file)
    if not isinstance(logs, list):
        raise ValueError(f"Inference log must contain a JSON list: {path}")
    return logs


def filter_logs_by_time(logs: Iterable[dict], days: int = DEFAULT_DAYS):
    """Keep log entries from the last ``days``, as in the deployment notebook."""
    if days < 1:
        raise ValueError("days must be at least 1.")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for entry in logs:
        if not isinstance(entry, dict):
            continue
        timestamp_value = entry.get("timestamp", "")
        if not timestamp_value:
            continue
        try:
            timestamp = datetime.fromisoformat(timestamp_value)
        except (TypeError, ValueError):
            filtered.append(entry)
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp >= cutoff:
            filtered.append(entry)
    return filtered


def calculate_confidence_metrics(confidences):
    values = np.asarray(confidences, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if not np.isfinite(values).all():
        raise ValueError("Confidence values must be finite.")
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def check_confidence_drift(current_confidences, baseline_stats):
    """Compare request confidence values with baseline confidence statistics."""
    default = {
        "ks_statistic": 0.0,
        "ks_pvalue": 1.0,
        "drift_detected": False,
        "threshold_ks": KS_PVALUE_THRESHOLD,
    }
    if not baseline_stats or "confidence_statistics" not in baseline_stats:
        return default

    current = np.asarray(current_confidences, dtype=float)
    if current.size < 2:
        return default
    if not np.isfinite(current).all():
        raise ValueError("Confidence values must be finite.")

    baseline = baseline_stats["confidence_statistics"]
    baseline_mean = float(baseline["mean"])
    baseline_std = float(baseline["std"])
    if baseline_std < 0:
        raise ValueError("Baseline confidence standard deviation cannot be negative.")
    current_mean = float(np.mean(current))
    baseline_samples = np.random.normal(
        baseline_mean,
        baseline_std,
        size=current.size,
    )
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current)
    mean_diff = abs(current_mean - baseline_mean)
    detected = bool(
        ks_pvalue < KS_PVALUE_THRESHOLD
        or mean_diff > MEAN_DRIFT_THRESHOLD
    )
    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "mean_diff": float(mean_diff),
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
        "drift_detected": detected,
        "threshold_ks": KS_PVALUE_THRESHOLD,
        "threshold_mean": MEAN_DRIFT_THRESHOLD,
    }


def check_class_distribution_drift(predictions, baseline_stats):
    """Compare predicted class labels to the baseline class proportions."""
    default = {
        "ks_statistic": 0.0,
        "ks_pvalue": 1.0,
        "drift_detected": False,
        "threshold_ks": KS_PVALUE_THRESHOLD,
    }
    if not baseline_stats or "class_distribution" not in baseline_stats:
        return default

    prediction_values = np.asarray(predictions, dtype=int)
    if prediction_values.size == 0:
        return default
    if not np.isin(prediction_values, (0, 1)).all():
        raise ValueError("Predictions must contain only class labels 0 and 1.")

    baseline = baseline_stats["class_distribution"]
    baseline_total = int(baseline["total"])
    if baseline_total <= 0:
        raise ValueError("Baseline class distribution must have a positive total.")
    class_0_count = int(baseline["class_0_count"])
    class_1_count = int(baseline["class_1_count"])
    if class_0_count < 0 or class_1_count < 0:
        raise ValueError("Baseline class counts cannot be negative.")
    if class_0_count + class_1_count != baseline_total:
        raise ValueError("Baseline class counts must sum to the baseline total.")

    baseline_distribution = np.array(
        [class_0_count / baseline_total, class_1_count / baseline_total]
    )
    baseline_samples = np.random.choice(
        [0, 1],
        size=baseline_total,
        p=baseline_distribution,
    )
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, prediction_values)
    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "drift_detected": bool(ks_pvalue < KS_PVALUE_THRESHOLD),
        "threshold_ks": KS_PVALUE_THRESHOLD,
    }


def detect_drift(
    logs,
    days: int = DEFAULT_DAYS,
    baseline_stats: Optional[dict] = None,
):
    """Return drift results with the keys used by the notebook's /drift route."""
    filtered_logs = filter_logs_by_time(logs, days=days)
    if not filtered_logs:
        return {
            "overall_drift_detected": False,
            "num_samples": 0,
            "confidence_drift": None,
            "class_distribution_drift": None,
            "error": "No logs found",
        }

    confidences = []
    predictions = []
    for entry in filtered_logs:
        confidence = entry.get("confidence")
        prediction = entry.get("prediction")
        if confidence is not None:
            confidences.append(float(confidence))
        if prediction is not None:
            predictions.append(int(prediction))

    baseline = baseline_stats or {}
    confidence_results = check_confidence_drift(confidences, baseline)
    class_results = check_class_distribution_drift(predictions, baseline)
    return {
        "overall_drift_detected": bool(
            confidence_results.get("drift_detected", False)
            or class_results.get("drift_detected", False)
        ),
        "num_samples": len(filtered_logs),
        "confidence_drift": confidence_results,
        "class_distribution_drift": class_results,
        "confidence_stats": calculate_confidence_metrics(confidences),
        "class_distribution": {
            "class_0_count": predictions.count(0),
            "class_1_count": predictions.count(1),
            "total": len(predictions),
        },
    }
