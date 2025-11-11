#!/usr/bin/env python3
"""
Detection manager orchestrating multiple anomaly/event detection strategies.
Provides a consistent interface for running baseline threshold detection,
PyOD-based outlier detection, and change-point detection with ruptures.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .event_detector import EventDetector

try:
    from pyod.models.iforest import IForest
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    PYOD_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PYOD_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RUPTURES_AVAILABLE = False


@dataclass
class DetectorResult:
    """Result container for a detector strategy."""

    detector_id: str
    detector_label: str
    detector_type: str
    events: List[Dict[str, Any]]
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "detector_id": self.detector_id,
            "detector_label": self.detector_label,
            "detector_type": self.detector_type,
            "events": self.events,
            "extras": self._serialize_extras(self.extras),
        }

    @staticmethod
    def _serialize_extras(extras: Dict[str, Any]) -> Dict[str, Any]:
        """Convert extras to JSON-serializable representation."""
        serialized: Dict[str, Any] = {}
        for key, value in extras.items():
            if isinstance(value, pd.Series):
                serialized[key] = {
                    "index": [str(idx) for idx in value.index],
                    "values": value.astype(float).tolist(),
                }
            elif isinstance(value, (np.ndarray, list, tuple)):
                serialized[key] = [float(v) if isinstance(v, (int, float, np.floating)) else v for v in value]
            else:
                serialized[key] = value
        return serialized


def deep_merge_dicts(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating the inputs."""
    if not overrides:
        return copy.deepcopy(base)

    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def calculate_severity(change_relative: float) -> str:
    """Calculate severity string based on relative change."""
    if change_relative < 0.05:
        return "low"
    if change_relative < 0.15:
        return "medium"
    return "high"


def find_continuous_segments(condition: pd.Series, max_gap: Optional[timedelta] = None) -> List[Tuple[int, int]]:
    """
    Find continuous segments where the boolean condition is True.
    Supports time-based gap merging similar to EventDetector._find_continuous_segments.
    """
    segments: List[Tuple[int, int]] = []
    in_segment = False
    segment_start: Optional[int] = None

    if max_gap is None and len(condition) > 1 and isinstance(condition.index, pd.DatetimeIndex):
        time_diffs = condition.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_interval = time_diffs.median()
            if median_interval >= timedelta(days=2):
                max_gap_candidate = median_interval * 3
                max_gap_default = timedelta(days=7)
                max_gap = max(max_gap_candidate, max_gap_default)
            else:
                max_gap_candidate = median_interval * 5
                max_gap_default = timedelta(days=1)
                max_gap = max(max_gap_candidate, max_gap_default)
    elif max_gap is None:
        max_gap = None

    for idx, value in enumerate(condition):
        if value and not in_segment:
            segment_start = idx
            in_segment = True
        elif not value and in_segment:
            segments.append((segment_start or 0, idx - 1))
            in_segment = False
        elif value and in_segment and max_gap is not None:
            if idx > (segment_start or 0):
                previous_time = condition.index[idx - 1]
                current_time = condition.index[idx]
                gap = current_time - previous_time
                if gap > max_gap:
                    segments.append((segment_start or 0, idx - 1))
                    segment_start = idx

    if in_segment:
        segments.append((segment_start or 0, len(condition) - 1))
    return segments


class BaseDetectorStrategy:
    """Abstract base class for detector strategies."""

    detector_type: str = "unknown"

    def __init__(
        self,
        detector_id: str,
        detector_label: str,
        global_config: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
    ):
        self.detector_id = detector_id
        self.detector_label = detector_label or detector_id
        self.global_config = copy.deepcopy(global_config)
        self.options = options or {}

        # Compute effective configs
        self.analytics_config = deep_merge_dicts(
            global_config.get("analytics", {}), self.options.get("analytics", {})
        )
        self.events_config = deep_merge_dicts(
            global_config.get("events", {}), self.options.get("events", {})
        )
        self.metric_direction_config = deep_merge_dicts(
            global_config.get("metric_direction", {}),
            self.options.get("metric_direction", {}),
        )

        self.detect_types = self.events_config.get("detect", ["degradation_start", "improvement_start"])
        self.hysteresis_points = self.analytics_config.get("hysteresis_points", 3)
        min_duration_minutes = self.events_config.get("min_event_duration_minutes", 30)
        self.min_event_duration = timedelta(minutes=min_duration_minutes)

    def detect(self, series: pd.Series, baseline_result: Dict[str, Any], metric_name: Optional[str] = None) -> DetectorResult:
        raise NotImplementedError

    # Helper utilities -----------------------------------------------------
    def get_metric_params(self, metric_name: Optional[str]) -> Tuple[float, float]:
        min_abs = self.analytics_config.get("min_absolute_change", 0.0)
        min_rel = self.analytics_config.get("min_relative_change", 0.0)
        metric_specific = self.analytics_config.get("metric_specific_params", {})
        if metric_name and metric_name in metric_specific:
            params = metric_specific[metric_name]
            min_abs = params.get("min_absolute_change", min_abs)
            min_rel = params.get("min_relative_change", min_rel)
        return float(min_abs), float(min_rel)

    def get_metric_direction(self, metric_name: Optional[str]) -> str:
        default_direction = self.metric_direction_config.get("default", "negative")
        if metric_name:
            return self.metric_direction_config.get(metric_name, default_direction)
        return default_direction

    def should_detect(self, event_type: str) -> bool:
        return event_type in self.detect_types

    def _segment_duration_ok(self, segment_start: pd.Timestamp, segment_end: pd.Timestamp, num_points: int) -> bool:
        if not isinstance(segment_start, pd.Timestamp) or not isinstance(segment_end, pd.Timestamp):
            return num_points >= self.hysteresis_points
        duration = segment_end - segment_start
        if duration >= self.min_event_duration:
            return True
        return num_points >= self.hysteresis_points * 2


class ThresholdDetectorStrategy(BaseDetectorStrategy):
    """Wrapper around the existing threshold-based EventDetector."""

    detector_type = "threshold"

    def __init__(
        self,
        detector_id: str,
        detector_label: str,
        global_config: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(detector_id, detector_label, global_config, options)
        detector_config = deep_merge_dicts(global_config, options or {})
        self.detector = EventDetector(detector_config)

    def detect(
        self,
        series: pd.Series,
        baseline_result: Dict[str, Any],
        metric_name: Optional[str] = None,
    ) -> DetectorResult:
        events = self.detector.detect_events(series, baseline_result, metric_name=metric_name)
        for event in events:
            event.setdefault("detector_id", self.detector_id)
            event.setdefault("detector_label", self.detector_label)
        extras = {
            "baseline_method": baseline_result.get("baseline_method"),
            "window_size": baseline_result.get("window_size"),
            "sensitivity": baseline_result.get("sensitivity"),
        }
        return DetectorResult(
            detector_id=self.detector_id,
            detector_label=self.detector_label,
            detector_type=self.detector_type,
            events=events,
            extras=extras,
        )


class PyODDetectorStrategy(BaseDetectorStrategy):
    """Detector strategy that leverages PyOD models for point anomaly detection."""

    detector_type = "pyod"

    MODEL_FACTORIES = {
        "iforest": lambda params: IForest(
            contamination=params.get("contamination", "auto"),
            n_estimators=params.get("n_estimators", 100),
            random_state=params.get("random_state"),
        ),
        "copod": lambda params: COPOD(),
        "ecod": lambda params: ECOD(),
    }

    def detect(
        self,
        series: pd.Series,
        baseline_result: Dict[str, Any],
        metric_name: Optional[str] = None,
    ) -> DetectorResult:
        if not PYOD_AVAILABLE:  # pragma: no cover - optional dependency
            warnings.warn("PyOD is not available; skipping PyOD detector.", RuntimeWarning)
            return DetectorResult(
                detector_id=self.detector_id,
                detector_label=self.detector_label,
                detector_type=self.detector_type,
                events=[],
                extras={"error": "pyod_not_available"},
            )

        params = self.options.get("params", {})
        model_name = params.get("model", "iforest").lower()
        model_factory = self.MODEL_FACTORIES.get(model_name, self.MODEL_FACTORIES["iforest"])
        model = model_factory(params)

        # Prepare data
        values = series.astype(float).values.reshape(-1, 1)
        valid_mask = ~np.isnan(values).flatten()
        if valid_mask.sum() <= max(params.get("min_points", 20), self.hysteresis_points * 2):
            return DetectorResult(
                detector_id=self.detector_id,
                detector_label=self.detector_label,
                detector_type=self.detector_type,
                events=[],
                extras={"warning": "insufficient_points"},
            )

        values_clean = values[valid_mask]
        index_clean = series.index[valid_mask]

        model.fit(values_clean)
        predictions = model.predict(values_clean)  # 1 = outlier
        scores = model.decision_function(values_clean)

        anomaly_mask = pd.Series(False, index=series.index)
        anomaly_mask.loc[index_clean[predictions == 1]] = True

        segments = find_continuous_segments(anomaly_mask)
        events: List[Dict[str, Any]] = []

        min_abs_change, min_rel_change = self.get_metric_params(metric_name)
        metric_direction = self.get_metric_direction(metric_name)
        baseline_series = baseline_result.get("baseline_series", pd.Series(dtype=float))
        severity_scores: List[float] = []

        for seg_start, seg_end in segments:
            segment_values = series.iloc[seg_start : seg_end + 1]
            start_time = segment_values.index[0]
            end_time = segment_values.index[-1]
            num_points = len(segment_values)

            if not self._segment_duration_ok(start_time, end_time, num_points):
                continue

            if baseline_series is not None and not baseline_series.empty and seg_start < len(baseline_series):
                baseline_value = float(baseline_series.iloc[seg_start])
            else:
                baseline_value = float(baseline_result.get("baseline_value", series.mean()))

            current_value = float(segment_values.mean())
            change_absolute = current_value - baseline_value
            change_relative = abs(change_absolute / baseline_value) if baseline_value != 0 else 0.0

            if abs(change_absolute) < min_abs_change or change_relative < min_rel_change:
                continue

            if metric_direction == "negative":
                event_type = "degradation_start" if change_absolute > 0 else "improvement_start"
            else:
                event_type = "improvement_start" if change_absolute > 0 else "degradation_start"

            if not self.should_detect(event_type):
                continue

            segment_scores = scores[(index_clean >= start_time) & (index_clean <= end_time)]
            severity = calculate_severity(change_relative)
            severity_scores.append(change_relative)

            event = {
                "event_type": event_type,
                "event_start_time": start_time,
                "event_end_time": end_time,
                "severity": severity,
                "baseline_before": baseline_value,
                "baseline_after": float(baseline_result.get("baseline_value", baseline_value)),
                "change_absolute": float(change_absolute),
                "change_relative": float(change_relative),
                "current_value": current_value,
                "detector_id": self.detector_id,
                "detector_label": self.detector_label,
                "detector_score_max": float(np.max(segment_scores)) if len(segment_scores) > 0 else None,
                "detector_score_mean": float(np.mean(segment_scores)) if len(segment_scores) > 0 else None,
            }
            events.append(event)

        extras = {
            "scores": pd.Series(scores, index=index_clean),
            "anomaly_fraction": float(predictions.sum() / len(predictions)) if len(predictions) > 0 else 0.0,
            "severity_scores": severity_scores,
        }

        return DetectorResult(
            detector_id=self.detector_id,
            detector_label=self.detector_label,
            detector_type=self.detector_type,
            events=events,
            extras=extras,
        )


class RupturesDetectorStrategy(BaseDetectorStrategy):
    """Change-point detector leveraging ruptures."""

    detector_type = "ruptures"

    def detect(
        self,
        series: pd.Series,
        baseline_result: Dict[str, Any],
        metric_name: Optional[str] = None,
    ) -> DetectorResult:
        if not RUPTURES_AVAILABLE:  # pragma: no cover - optional dependency
            warnings.warn("ruptures is not available; skipping change-point detector.", RuntimeWarning)
            return DetectorResult(
                detector_id=self.detector_id,
                detector_label=self.detector_label,
                detector_type=self.detector_type,
                events=[],
                extras={"error": "ruptures_not_available"},
            )

        params = self.options.get("params", {})
        min_points = params.get("min_points", max(2 * self.hysteresis_points, 30))
        if len(series) < min_points:
            return DetectorResult(
                detector_id=self.detector_id,
                detector_label=self.detector_label,
                detector_type=self.detector_type,
                events=[],
                extras={"warning": "insufficient_points"},
            )

        model_name = params.get("model", "rbf")
        penalty = params.get("penalty", params.get("pen", 10))
        n_bkps = params.get("n_bkps")
        window = params.get("window", max(self.hysteresis_points * 2, 10))

        signal = series.astype(float).values
        algo = rpt.Binseg(model=model_name).fit(signal)

        if n_bkps:
            breakpoints = algo.predict(n_bkps=int(n_bkps))
        else:
            breakpoints = algo.predict(pen=int(penalty))

        # Remove trivial breakpoint at end
        breakpoints = [bp for bp in breakpoints if bp < len(signal)]

        min_abs_change, min_rel_change = self.get_metric_params(metric_name)
        metric_direction = self.get_metric_direction(metric_name)
        baseline_series = baseline_result.get("baseline_series", pd.Series(dtype=float))
        events: List[Dict[str, Any]] = []

        for bp in breakpoints:
            left_start = max(0, bp - window)
            left_end = bp
            right_start = bp
            right_end = min(len(signal), bp + window)

            if left_end - left_start < self.hysteresis_points or right_end - right_start < self.hysteresis_points:
                continue

            mean_before = float(np.mean(signal[left_start:left_end]))
            mean_after = float(np.mean(signal[right_start:right_end]))
            change_absolute = mean_after - mean_before
            change_relative = abs(change_absolute / mean_before) if mean_before != 0 else 0.0

            if abs(change_absolute) < min_abs_change or change_relative < min_rel_change:
                continue

            if metric_direction == "negative":
                direction = "degradation_start" if change_absolute > 0 else "improvement_start"
            else:
                direction = "improvement_start" if change_absolute > 0 else "degradation_start"

            event_type = "threshold_shift" if self.should_detect("threshold_shift") else direction
            if not self.should_detect(event_type):
                continue

            start_time = series.index[left_start]
            end_time = series.index[right_end - 1]

            if not self._segment_duration_ok(start_time, end_time, right_end - left_start):
                continue

            if baseline_series is not None and not baseline_series.empty and left_start < len(baseline_series):
                baseline_before = float(baseline_series.iloc[left_start])
            else:
                baseline_before = mean_before

            event = {
                "event_type": event_type,
                "event_start_time": start_time,
                "event_end_time": end_time,
                "severity": calculate_severity(change_relative),
                "baseline_before": baseline_before,
                "baseline_after": mean_after,
                "change_absolute": float(change_absolute),
                "change_relative": float(change_relative),
                "current_value": mean_after,
                "detector_id": self.detector_id,
                "detector_label": self.detector_label,
            }
            events.append(event)

        extras = {
            "breakpoints": breakpoints,
            "penalty": penalty,
            "window": window,
        }

        return DetectorResult(
            detector_id=self.detector_id,
            detector_label=self.detector_label,
            detector_type=self.detector_type,
            events=events,
            extras=extras,
        )


STRATEGY_FACTORY = {
    "threshold": ThresholdDetectorStrategy,
    "baseline_threshold": ThresholdDetectorStrategy,
    "pyod": PyODDetectorStrategy,
    "pyod_iforest": PyODDetectorStrategy,
    "pyod_copod": PyODDetectorStrategy,
    "pyod_ecod": PyODDetectorStrategy,
    "ruptures": RupturesDetectorStrategy,
    "change_point": RupturesDetectorStrategy,
}


class DetectionManager:
    """Manage multiple detector strategies and orchestrate detection runs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.events_config = config.get("events", {})
        self.detector_specs = self._parse_detector_specs(self.events_config)

        self.detectors: Dict[str, BaseDetectorStrategy] = {}
        for spec in self.detector_specs["ordered_specs"]:
            strategy = self._build_detector(spec)
            if strategy:
                self.detectors[spec["id"]] = strategy

        # Ensure at least baseline detector exists
        if not self.detectors:
            default_spec = {
                "id": "baseline_threshold",
                "type": "threshold",
                "label": "Baseline Threshold",
                "params": {},
                "options": {},
            }
            self.detectors["baseline_threshold"] = ThresholdDetectorStrategy(
                "baseline_threshold", "Baseline Threshold", self.config, {}
            )
            self.detector_specs["default_id"] = "baseline_threshold"

    # ------------------------------------------------------------------ public API
    def detect_all(
        self,
        series: pd.Series,
        baseline_result: Dict[str, Any],
        metric_name: Optional[str] = None,
    ) -> List[DetectorResult]:
        results: List[DetectorResult] = []
        for detector_id in self.detector_specs["order"]:
            strategy = self.detectors.get(detector_id)
            if not strategy:
                continue
            result = strategy.detect(series, baseline_result, metric_name=metric_name)
            results.append(result)
        return results

    def detect_default(
        self,
        series: pd.Series,
        baseline_result: Dict[str, Any],
        metric_name: Optional[str] = None,
    ) -> DetectorResult:
        default_id = self.detector_specs.get("default_id")
        if default_id and default_id in self.detectors:
            return self.detectors[default_id].detect(series, baseline_result, metric_name=metric_name)
        # Fallback to first detector
        for detector in self.detectors.values():
            return detector.detect(series, baseline_result, metric_name=metric_name)
        return DetectorResult(
            detector_id="baseline_threshold",
            detector_label="Baseline Threshold",
            detector_type="threshold",
            events=[],
            extras={},
        )

    def get_default_detector_id(self) -> str:
        return self.detector_specs.get("default_id", next(iter(self.detectors.keys())))

    def get_compare_detector_ids(self) -> List[str]:
        compare = self.detector_specs.get("compare_ids", [])
        return [det_id for det_id in compare if det_id in self.detectors]

    # ------------------------------------------------------------------ internal helpers
    def _parse_detector_specs(self, events_config: Dict[str, Any]) -> Dict[str, Any]:
        detectors_cfg = events_config.get("detectors")
        default_id = events_config.get("default_detector")
        compare_ids = events_config.get("compare_detectors", [])

        specs: Dict[str, Dict[str, Any]] = {}

        if isinstance(detectors_cfg, dict):
            for det_id, det_cfg in detectors_cfg.items():
                specs[det_id] = self._normalize_detector_config(det_id, det_cfg)
        elif isinstance(detectors_cfg, list):
            for det_cfg in detectors_cfg:
                det_id = det_cfg.get("id") or det_cfg.get("name")
                if not det_id:
                    continue
                specs[det_id] = self._normalize_detector_config(det_id, det_cfg)
        else:
            specs["baseline_threshold"] = {
                "id": "baseline_threshold",
                "type": "threshold",
                "label": "Baseline Threshold",
                "params": {},
                "options": {},
                "enabled": True,
            }

        if not default_id or default_id not in specs:
            default_id = next(iter(specs.keys()))

        order = list(specs.keys())
        ordered_specs = [specs[det_id] for det_id in order]

        return {
            "specs": specs,
            "order": order,
            "ordered_specs": ordered_specs,
            "default_id": default_id,
            "compare_ids": compare_ids,
        }

    def _normalize_detector_config(self, det_id: str, det_cfg: Dict[str, Any]) -> Dict[str, Any]:
        detector_type = det_cfg.get("type", det_cfg.get("strategy", "threshold"))
        label = det_cfg.get("label") or det_cfg.get("name") or det_id.replace("_", " ").title()
        params = det_cfg.get("params", {})
        options = {k: v for k, v in det_cfg.items() if k in {"analytics", "events", "metric_direction"}}
        enabled = det_cfg.get("enabled", True)
        return {
            "id": det_id,
            "type": detector_type,
            "label": label,
            "params": params,
            "options": options,
            "enabled": enabled,
        }

    def _build_detector(self, spec: Dict[str, Any]) -> Optional[BaseDetectorStrategy]:
        if not spec.get("enabled", True):
            return None

        detector_type = spec.get("type", "threshold")
        factory = STRATEGY_FACTORY.get(detector_type)
        if factory is None:
            warnings.warn(f"Unknown detector type '{detector_type}', skipping.", RuntimeWarning)
            return None

        # Special-case PyOD variations
        options = spec.get("options", {})
        params = spec.get("params", {})
        if detector_type in {"pyod_iforest", "pyod_copod", "pyod_ecod"}:
            options = copy.deepcopy(options)
            options.setdefault("params", {}).update(params)
            options["params"]["model"] = detector_type.split("_", 1)[-1]
        else:
            options = copy.deepcopy(options)
            options.setdefault("params", {}).update(params)

        try:
            return factory(spec["id"], spec["label"], self.config, options)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Failed to initialize detector '{spec['id']}': {exc}", RuntimeWarning)
            return None

