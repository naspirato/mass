#!/usr/bin/env python3
"""
Tests for the DetectionManager orchestrator supporting multiple detector strategies.
"""

import copy
import unittest
from datetime import datetime, timedelta

import pandas as pd

from mass.core.baseline_calculator import BaselineCalculator
from mass.core.detection_manager import DetectionManager, PYOD_AVAILABLE, RUPTURES_AVAILABLE


class TestDetectionManager(unittest.TestCase):
    """Unit tests for DetectionManager."""

    def setUp(self):
        base_time = datetime.now() - timedelta(days=5)
        self.timestamps = pd.date_range(start=base_time, periods=80, freq='1h')

        self.config = {
            'analytics': {
                'baseline_method': 'rolling_mean',
                'window_size': 12,
                'sensitivity': 2.0,
                'min_absolute_change': 5,
                'min_relative_change': 0.05,
                'hysteresis_points': 3,
                'adaptive_threshold': True
            },
            'events': {
                'detect': ['degradation_start', 'improvement_start'],
                'min_event_duration_minutes': 30,
                'detectors': {
                    'baseline_threshold': {
                        'type': 'threshold',
                        'label': 'Baseline Threshold'
                    },
                    'pyod_iforest': {
                        'type': 'pyod_iforest',
                        'label': 'PyOD Isolation Forest',
                        'params': {
                            'contamination': 0.1,
                            'min_points': 15,
                            'random_state': 42
                        }
                    }
                },
                'default_detector': 'baseline_threshold',
                'compare_detectors': ['baseline_threshold', 'pyod_iforest']
            },
            'metric_direction': {
                'default': 'negative'
            },
            'context_fields': ['operation_type'],
            'metric_fields': ['metric_name', 'metric_value'],
            'timestamp_field': 'ts'
        }

    def _build_series(self):
        """Create time series with a clear degradation event."""
        values = [100.0] * 50 + [180.0] * 30
        series = pd.Series(values, index=self.timestamps)
        return series

    def test_detect_all_returns_results_with_metadata(self):
        series = self._build_series()
        baseline_calc = BaselineCalculator(self.config)
        baseline_result = baseline_calc.compute_baseline_and_thresholds(series)

        manager = DetectionManager(self.config)
        results = manager.detect_all(series, baseline_result, metric_name='duration_ms')

        self.assertGreaterEqual(len(results), 1)
        detector_ids = [result.detector_id for result in results]
        self.assertIn('baseline_threshold', detector_ids)
        if PYOD_AVAILABLE:
            self.assertIn('pyod_iforest', detector_ids)

        default_result = manager.detect_default(series, baseline_result, metric_name='duration_ms')
        self.assertEqual(default_result.detector_id, manager.get_default_detector_id())

        if default_result.events:
            for event in default_result.events:
                self.assertEqual(event.get('detector_id'), default_result.detector_id)
                self.assertEqual(event.get('detector_label'), default_result.detector_label)

    @unittest.skipUnless(PYOD_AVAILABLE, "PyOD is not installed")
    def test_pyod_detector_reports_degradation(self):
        config = copy.deepcopy(self.config)
        config['analytics']['hysteresis_points'] = 1
        config['events']['min_event_duration_minutes'] = 0
        config['events']['detectors']['pyod_iforest']['analytics'] = {
            'hysteresis_points': 1,
            'min_absolute_change': 1,
            'min_relative_change': 0.01
        }
        config['events']['detectors']['pyod_iforest']['events'] = {
            'min_event_duration_minutes': 0
        }
        config['events']['detectors']['pyod_iforest']['params']['contamination'] = 0.3

        values = [100.0] * 40 + [400.0] * 20
        index = self.timestamps[:60]
        series = pd.Series(values, index=index)
        baseline_series = pd.Series([100.0] * len(series), index=index)
        baseline_result = {
            'baseline_series': baseline_series,
            'baseline_value': 100.0,
            'upper_threshold': 150.0,
            'lower_threshold': 50.0,
            'sensitivity': 2.0,
            'adaptive_threshold': False,
            'baseline_method': 'constant',
            'window_size': 1
        }

        manager = DetectionManager(config)
        results = manager.detect_all(series, baseline_result, metric_name='duration_ms')

        pyod_result = next((res for res in results if res.detector_id == 'pyod_iforest'), None)
        self.assertIsNotNone(pyod_result, "PyOD detector result not found")
        self.assertIsNotNone(pyod_result.extras.get('scores'))

    def test_disabled_detectors_are_ignored_but_default_falls_back(self):
        series = self._build_series()
        baseline_calc = BaselineCalculator(self.config)
        baseline_result = baseline_calc.compute_baseline_and_thresholds(series)

        config = copy.deepcopy(self.config)
        events_cfg = config['events']
        detectors_cfg = events_cfg['detectors']
        detectors_cfg['pyod_iforest'] = dict(detectors_cfg['pyod_iforest'])
        detectors_cfg['pyod_iforest']['enabled'] = False
        events_cfg['detectors'] = detectors_cfg
        events_cfg['default_detector'] = 'pyod_iforest'
        events_cfg['compare_detectors'] = ['baseline_threshold', 'pyod_iforest']
        config['events'] = events_cfg

        manager = DetectionManager(config)

        self.assertEqual(set(manager.detectors.keys()), {'baseline_threshold'})
        self.assertEqual(manager.get_compare_detector_ids(), ['baseline_threshold'])

        default_result = manager.detect_default(series, baseline_result, metric_name='duration_ms')
        self.assertEqual(default_result.detector_id, 'baseline_threshold')

    def test_list_form_detectors_builds_and_defaults(self):
        series = self._build_series()
        baseline_calc = BaselineCalculator(self.config)
        baseline_result = baseline_calc.compute_baseline_and_thresholds(series)

        config = copy.deepcopy(self.config)
        config['events']['detectors'] = [
            {'id': 'baseline_threshold', 'type': 'threshold', 'label': 'Baseline Threshold'},
            {'id': 'ruptures_binseg', 'type': 'ruptures', 'enabled': False}
        ]
        config['events'].pop('default_detector', None)
        config['events']['compare_detectors'] = ['baseline_threshold', 'ruptures_binseg']

        manager = DetectionManager(config)

        self.assertEqual(manager.get_default_detector_id(), 'baseline_threshold')
        self.assertEqual(set(manager.detectors.keys()), {'baseline_threshold'})

        results = manager.detect_all(series, baseline_result, metric_name='duration_ms')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].detector_id, 'baseline_threshold')

    @unittest.skipUnless(RUPTURES_AVAILABLE, "ruptures is not installed")
    def test_ruptures_detector_reports_shift(self):
        series = self._build_series()
        baseline_calc = BaselineCalculator(self.config)
        baseline_result = baseline_calc.compute_baseline_and_thresholds(series)

        config = copy.deepcopy(self.config)
        config['events']['detect'] = ['degradation_start', 'improvement_start', 'threshold_shift']
        config['events']['detectors']['ruptures_binseg'] = {
            'type': 'ruptures',
            'label': 'Ruptures Binseg',
            'params': {
                'n_bkps': 1,
                'penalty': 5,
                'window': 10
            }
        }
        config['events']['compare_detectors'] = ['baseline_threshold', 'ruptures_binseg']

        manager = DetectionManager(config)
        results = manager.detect_all(series, baseline_result, metric_name='duration_ms')
        ruptures_result = next((res for res in results if res.detector_id == 'ruptures_binseg'), None)

        self.assertIsNotNone(ruptures_result, "Ruptures detector result not found")
        self.assertGreater(len(ruptures_result.events), 0, "Ruptures detector should report shifts")
        event_types = {event['event_type'] for event in ruptures_result.events}
        self.assertIn('threshold_shift', event_types)


if __name__ == '__main__':
    unittest.main()
