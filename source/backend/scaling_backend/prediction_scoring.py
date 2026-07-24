from __future__ import annotations

import math
from typing import Any


PREDICTION_INTERVAL_ALPHA = 0.2
PREDICTION_INTERVAL_MISS_MULTIPLIER = 2 / PREDICTION_INTERVAL_ALPHA
PREDICTION_POINT_ERROR_WEIGHT = 0.5
PREDICTION_INTERVAL_SCORE_WEIGHT = 0.5


def compute_prediction_metrics(
    *,
    predicted_final_loss: Any,
    predicted_final_loss_lower: Any,
    predicted_final_loss_upper: Any,
    actual_final_validation_loss: Any,
) -> dict[str, Any]:
    actual = _optional_finite_float(actual_final_validation_loss)
    predicted = _optional_finite_float(predicted_final_loss)
    lower = _optional_finite_float(predicted_final_loss_lower)
    upper = _optional_finite_float(predicted_final_loss_upper)

    metrics = {
        "prediction_absolute_error": None,
        "prediction_interval_covered": None,
        "prediction_interval_width": None,
        "prediction_interval_miss_distance": None,
        "prediction_interval_score": None,
        "prediction_quality_penalty": None,
    }
    if actual is None or predicted is None:
        return metrics

    point_error = abs(predicted - actual)
    metrics["prediction_absolute_error"] = _round_metric(point_error)

    if lower is None or upper is None or lower > upper:
        return metrics

    interval_width = upper - lower
    miss_distance = max(0.0, lower - actual) + max(0.0, actual - upper)
    interval_score = (
        interval_width + PREDICTION_INTERVAL_MISS_MULTIPLIER * miss_distance
    )
    prediction_quality_penalty = (
        PREDICTION_POINT_ERROR_WEIGHT * point_error
        + PREDICTION_INTERVAL_SCORE_WEIGHT * interval_score
    )

    metrics.update(
        {
            "prediction_interval_covered": lower <= actual <= upper,
            "prediction_interval_width": _round_metric(interval_width),
            "prediction_interval_miss_distance": _round_metric(miss_distance),
            "prediction_interval_score": _round_metric(interval_score),
            "prediction_quality_penalty": _round_metric(prediction_quality_penalty),
        }
    )
    return metrics


def _optional_finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _round_metric(value: float) -> float:
    return round(value, 12)
